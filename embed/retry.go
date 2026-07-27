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
	"strconv"
	"time"

	"github.com/conduitio/conduit-processor-sdk/egress"
	"github.com/jpillora/backoff"
)

// retryConfig bounds doWithRetry's behavior. Built from Config's
// RetryBackoff*/MaxRetries fields.
type retryConfig struct {
	Min        time.Duration
	Max        time.Duration
	Factor     float64
	MaxRetries int
}

// retryableEgressError reports whether an [egress.Do] error is transient
// infrastructure trouble worth retrying (a timeout, DNS hiccup, or
// transport reset) as opposed to a policy or shape rejection
// (ErrForbidden, ErrEgressDisabled, ErrInvalidRequest,
// ErrResponseTooLarge) that will not resolve by retrying — see the egress
// package's doc.go for what each sentinel means.
func retryableEgressError(err error) bool {
	return errors.Is(err, egress.ErrTimeout) ||
		errors.Is(err, egress.ErrDNS) ||
		errors.Is(err, egress.ErrTransport)
}

// retryableStatus reports whether an HTTP response status is worth
// retrying: 429 (rate limit, design doc §2/§7) or a 5xx server error.
// Any other status — including every other 4xx, e.g. 401 auth failure or
// 400 bad request — is returned to the caller immediately, unretried, so a
// misconfigured pipeline fails fast instead of burning its retry budget
// (design doc Failure Modes §1).
func retryableStatus(code int) bool {
	return code == http.StatusTooManyRequests || code >= 500
}

// retryAfter parses a Retry-After response header as a duration, honoring
// only the delta-seconds form (the HTTP-date form is not needed for the
// providers this package targets). Returns 0, false if absent or
// unparsable, in which case the caller falls back to its own exponential
// backoff.
func retryAfter(resp egress.Response) (time.Duration, bool) {
	for _, v := range resp.Headers["Retry-After"] {
		if secs, err := strconv.Atoi(v); err == nil && secs >= 0 {
			return time.Duration(secs) * time.Second, true
		}
	}
	return 0, false
}

// doWithRetry performs one host-mediated egress call, retrying on a
// retryable status ([retryableStatus]) or a retryable transient egress
// error ([retryableEgressError]), honoring a Retry-After header when the
// provider sends one, otherwise backing off exponentially
// (jpillora/backoff, mirroring the core engine's built-in cohere.embed
// processor). Bounded by cfg.MaxRetries; the caller is responsible for
// turning an exhausted-retries outcome (a non-nil error, or a response
// whose status is still retryable when this returns) into a coded
// ai.embedding_provider_error.
//
// doWithRetry never retries across a sub-batch boundary and never invents
// a partial result: it always returns exactly what the final attempt
// produced.
func doWithRetry(ctx context.Context, cfg retryConfig, call func(ctx context.Context) (egress.Response, error)) (egress.Response, error) {
	b := &backoff.Backoff{Min: cfg.Min, Max: cfg.Max, Factor: cfg.Factor}

	for attempt := 0; ; attempt++ {
		resp, err := call(ctx)

		retryable := false
		var wait time.Duration
		switch {
		case err != nil:
			retryable = retryableEgressError(err)
		case retryableStatus(resp.StatusCode):
			retryable = true
			if d, ok := retryAfter(resp); ok {
				wait = d
			}
		}

		if !retryable || attempt >= cfg.MaxRetries {
			return resp, err
		}

		if wait <= 0 {
			wait = b.Duration()
		}

		select {
		case <-ctx.Done():
			return egress.Response{}, ctx.Err()
		case <-time.After(wait):
		}
	}
}
