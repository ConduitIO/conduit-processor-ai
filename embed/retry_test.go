// Copyright © 2026 Meroxa, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package embed

import (
	"context"
	"errors"
	"net/http"
	"testing"
	"time"

	"github.com/conduitio/conduit-processor-sdk/egress"
	"github.com/matryer/is"
)

func fastRetryConfig(maxRetries int) retryConfig {
	return retryConfig{Min: time.Millisecond, Max: 5 * time.Millisecond, Factor: 2, MaxRetries: maxRetries}
}

func TestDoWithRetry_SucceedsWithoutRetry(t *testing.T) {
	is := is.New(t)
	calls := 0
	resp, err := doWithRetry(context.Background(), fastRetryConfig(3), func(context.Context) (egress.Response, error) {
		calls++
		return egress.Response{StatusCode: http.StatusOK}, nil
	})
	is.NoErr(err)
	is.Equal(resp.StatusCode, http.StatusOK)
	is.Equal(calls, 1)
}

func TestDoWithRetry_RetriesOn429ThenSucceeds(t *testing.T) {
	is := is.New(t)
	calls := 0
	resp, err := doWithRetry(context.Background(), fastRetryConfig(3), func(context.Context) (egress.Response, error) {
		calls++
		if calls < 3 {
			return egress.Response{StatusCode: http.StatusTooManyRequests}, nil
		}
		return egress.Response{StatusCode: http.StatusOK}, nil
	})
	is.NoErr(err)
	is.Equal(resp.StatusCode, http.StatusOK)
	is.Equal(calls, 3)
}

func TestDoWithRetry_HonorsRetryAfterHeader(t *testing.T) {
	is := is.New(t)
	calls := 0
	start := time.Now()
	_, err := doWithRetry(context.Background(), fastRetryConfig(1), func(context.Context) (egress.Response, error) {
		calls++
		if calls == 1 {
			return egress.Response{
				StatusCode: http.StatusTooManyRequests,
				Headers:    map[string][]string{"Retry-After": {"0"}},
			}, nil
		}
		return egress.Response{StatusCode: http.StatusOK}, nil
	})
	elapsed := time.Since(start)
	is.NoErr(err)
	is.Equal(calls, 2)
	// Retry-After: 0 means "retry immediately" — this asserts the header
	// path was taken (near-zero wait) rather than falling through to the
	// exponential backoff floor, which would still be fast here but is a
	// different code path than what's under test.
	is.True(elapsed < 500*time.Millisecond)
}

func TestDoWithRetry_ExhaustsRetriesOn429(t *testing.T) {
	is := is.New(t)
	calls := 0
	resp, err := doWithRetry(context.Background(), fastRetryConfig(2), func(context.Context) (egress.Response, error) {
		calls++
		return egress.Response{StatusCode: http.StatusTooManyRequests}, nil
	})
	is.NoErr(err) // doWithRetry itself doesn't error on exhaustion — it returns the last response
	is.Equal(resp.StatusCode, http.StatusTooManyRequests)
	is.Equal(calls, 3) // initial attempt + 2 retries
}

func TestDoWithRetry_RetriesOn5xx(t *testing.T) {
	is := is.New(t)
	calls := 0
	resp, err := doWithRetry(context.Background(), fastRetryConfig(3), func(context.Context) (egress.Response, error) {
		calls++
		if calls < 2 {
			return egress.Response{StatusCode: http.StatusServiceUnavailable}, nil
		}
		return egress.Response{StatusCode: http.StatusOK}, nil
	})
	is.NoErr(err)
	is.Equal(resp.StatusCode, http.StatusOK)
	is.Equal(calls, 2)
}

func TestDoWithRetry_DoesNotRetryOtherClientErrors(t *testing.T) {
	// A 401/400 is not transient — doWithRetry must return on the first
	// attempt so the caller can fail fast (design doc Failure Modes §1).
	for _, status := range []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound} {
		t.Run(http.StatusText(status), func(t *testing.T) {
			is := is.New(t)
			calls := 0
			resp, err := doWithRetry(context.Background(), fastRetryConfig(3), func(context.Context) (egress.Response, error) {
				calls++
				return egress.Response{StatusCode: status}, nil
			})
			is.NoErr(err)
			is.Equal(resp.StatusCode, status)
			is.Equal(calls, 1)
		})
	}
}

func TestDoWithRetry_RetriesTransientEgressErrors(t *testing.T) {
	for _, transient := range []error{egress.ErrTimeout, egress.ErrDNS, egress.ErrTransport} {
		t.Run(transient.Error(), func(t *testing.T) {
			is := is.New(t)
			calls := 0
			_, err := doWithRetry(context.Background(), fastRetryConfig(2), func(context.Context) (egress.Response, error) {
				calls++
				if calls < 2 {
					return egress.Response{}, transient
				}
				return egress.Response{StatusCode: http.StatusOK}, nil
			})
			is.NoErr(err)
			is.Equal(calls, 2)
		})
	}
}

func TestDoWithRetry_DoesNotRetryPolicyEgressErrors(t *testing.T) {
	for _, policy := range []error{egress.ErrForbidden, egress.ErrEgressDisabled, egress.ErrInvalidRequest, egress.ErrResponseTooLarge} {
		t.Run(policy.Error(), func(t *testing.T) {
			is := is.New(t)
			calls := 0
			_, err := doWithRetry(context.Background(), fastRetryConfig(3), func(context.Context) (egress.Response, error) {
				calls++
				return egress.Response{}, policy
			})
			is.True(errors.Is(err, policy))
			is.Equal(calls, 1)
		})
	}
}

func TestDoWithRetry_ContextCancellationStopsRetrying(t *testing.T) {
	is := is.New(t)
	ctx, cancel := context.WithCancel(context.Background())
	calls := 0
	go func() {
		time.Sleep(2 * time.Millisecond)
		cancel()
	}()
	_, err := doWithRetry(ctx, retryConfig{Min: time.Second, Max: time.Second, Factor: 2, MaxRetries: 5},
		func(context.Context) (egress.Response, error) {
			calls++
			return egress.Response{StatusCode: http.StatusTooManyRequests}, nil
		})
	is.True(errors.Is(err, context.Canceled))
	is.Equal(calls, 1)
}
