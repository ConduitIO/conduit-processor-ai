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
	"encoding/json"
	"errors"
	"net/http"
	"testing"
	"time"

	"github.com/conduitio/conduit-processor-sdk/egress"
	"github.com/conduitio/conduit-processor-sdk/pprocutils"
	"github.com/conduitio/conduit-processor-sdk/pprocutils/mock"
	"github.com/matryer/is"
	"go.uber.org/mock/gomock"
)

// withHTTPService swaps the package-level egress.HTTPService for the
// duration of a test — the same seam the SDK's own egress package tests
// use to inject a fake transport ("fake egress.Do") without a real WASM
// host. It restores the disabled default afterwards.
func withHTTPService(t *testing.T, svc pprocutils.HTTPService) {
	t.Helper()
	egress.HTTPService = svc
	t.Cleanup(func() { egress.HTTPService = disabledService{} })
}

// disabledService mirrors egress's own unexported deny-all default so
// tests can restore it without importing egress's internals.
type disabledService struct{}

func (disabledService) Do(context.Context, pprocutils.HTTPRequest) (pprocutils.HTTPResponse, error) {
	return pprocutils.HTTPResponse{}, egress.ErrEgressDisabled
}

func testOpenAIConfig() Config {
	return Config{
		OpenAIAuthSecretRef: "openai-key",
		OpenAIBaseURL:       "https://api.openai.com",
		Model:               "text-embedding-3-small",
		RequestTimeout:      time.Second,
		MaxRetries:          2,
		RetryBackoffMin:     time.Millisecond,
		RetryBackoffMax:     5 * time.Millisecond,
		RetryBackoffFactor:  2,
	}
}

func openAISuccessBody(t *testing.T, dims map[int]int, totalTokens int) []byte {
	t.Helper()
	type datum struct {
		Index     int       `json:"index"`
		Embedding []float32 `json:"embedding"`
	}
	resp := struct {
		Data  []datum `json:"data"`
		Model string  `json:"model"`
		Usage struct {
			TotalTokens int `json:"total_tokens"`
		} `json:"usage"`
	}{Model: "text-embedding-3-small"}
	resp.Usage.TotalTokens = totalTokens

	for idx, dim := range dims {
		vec := make([]float32, dim)
		for i := range vec {
			vec[i] = float32(i) * 0.01
		}
		resp.Data = append(resp.Data, datum{Index: idx, Embedding: vec})
	}
	body, err := json.Marshal(resp)
	if err != nil {
		t.Fatalf("marshal fixture: %v", err)
	}
	return body
}

func TestOpenAIProvider_EmbedSuccess_RequestResponseRoundTrip(t *testing.T) {
	is := is.New(t)
	ctrl := gomock.NewController(t)
	svc := mock.NewHTTPService(ctrl)
	withHTTPService(t, svc)

	var captured pprocutils.HTTPRequest
	svc.EXPECT().Do(gomock.Any(), gomock.Any()).DoAndReturn(
		func(_ context.Context, req pprocutils.HTTPRequest) (pprocutils.HTTPResponse, error) {
			captured = req
			return pprocutils.HTTPResponse{
				StatusCode: http.StatusOK,
				Body:       openAISuccessBody(t, map[int]int{0: 3, 1: 3}, 42),
			}, nil
		})

	p, err := newOpenAIProvider(testOpenAIConfig())
	is.NoErr(err)

	result, err := p.Embed(context.Background(), []string{"hello", "world"})
	is.NoErr(err)

	// Request round-trip: method, URL, auth secret ref (never a raw
	// Authorization header — that's the host's job), and the marshaled
	// input array.
	is.Equal(captured.Method, http.MethodPost)
	is.Equal(captured.URL, "https://api.openai.com/v1/embeddings")
	is.Equal(captured.AuthSecretRef, "openai-key")
	var reqBody openAIEmbeddingRequest
	is.NoErr(json.Unmarshal(captured.Body, &reqBody))
	is.Equal(reqBody.Input, []string{"hello", "world"})
	is.Equal(reqBody.Model, "text-embedding-3-small")

	// Response round-trip: 1:1 outcomes in input order, dimension, batch-
	// scoped tokens (OpenAI reports usage per call, not per input).
	is.Equal(len(result.Outcomes), 2)
	is.Equal(len(result.Outcomes[0].Vector), 3)
	is.Equal(len(result.Outcomes[1].Vector), 3)
	is.NoErr(result.Outcomes[0].Err)
	is.Equal(result.Dimension, 3)
	is.Equal(result.TokensUsed, 42)
	is.Equal(result.TokensScope, TokensScopeBatch)
}

func TestOpenAIProvider_SingleInputTokensScopeIsRecord(t *testing.T) {
	is := is.New(t)
	ctrl := gomock.NewController(t)
	svc := mock.NewHTTPService(ctrl)
	withHTTPService(t, svc)

	svc.EXPECT().Do(gomock.Any(), gomock.Any()).Return(pprocutils.HTTPResponse{
		StatusCode: http.StatusOK,
		Body:       openAISuccessBody(t, map[int]int{0: 3}, 7),
	}, nil)

	p, err := newOpenAIProvider(testOpenAIConfig())
	is.NoErr(err)
	result, err := p.Embed(context.Background(), []string{"solo"})
	is.NoErr(err)
	is.Equal(result.TokensScope, TokensScopeRecord)
	is.Equal(result.TokensUsed, 7)
}

func TestOpenAIProvider_FullBatchFailure_AuthError_NotRetried(t *testing.T) {
	is := is.New(t)
	ctrl := gomock.NewController(t)
	svc := mock.NewHTTPService(ctrl)
	withHTTPService(t, svc)

	calls := 0
	svc.EXPECT().Do(gomock.Any(), gomock.Any()).DoAndReturn(
		func(context.Context, pprocutils.HTTPRequest) (pprocutils.HTTPResponse, error) {
			calls++
			return pprocutils.HTTPResponse{
				StatusCode: http.StatusUnauthorized,
				Body:       []byte(`{"error":{"message":"Incorrect API key provided","type":"invalid_request_error"}}`),
			}, nil
		}).Times(1)

	p, err := newOpenAIProvider(testOpenAIConfig())
	is.NoErr(err)
	_, err = p.Embed(context.Background(), []string{"a", "b"})
	is.True(err != nil)
	var perr *Error
	is.True(errors.As(err, &perr))
	is.Equal(perr.Code, CodeProviderError)
	is.Equal(calls, 1) // gomock.Times(1) would fail the test if a retry happened
}

func TestOpenAIProvider_RateLimitExhaustion_ReturnsProviderError(t *testing.T) {
	is := is.New(t)
	ctrl := gomock.NewController(t)
	svc := mock.NewHTTPService(ctrl)
	withHTTPService(t, svc)

	calls := 0
	svc.EXPECT().Do(gomock.Any(), gomock.Any()).DoAndReturn(
		func(context.Context, pprocutils.HTTPRequest) (pprocutils.HTTPResponse, error) {
			calls++
			return pprocutils.HTTPResponse{StatusCode: http.StatusTooManyRequests}, nil
		}).MinTimes(1)

	cfg := testOpenAIConfig()
	cfg.MaxRetries = 2
	p, err := newOpenAIProvider(cfg)
	is.NoErr(err)

	_, err = p.Embed(context.Background(), []string{"a"})
	is.True(err != nil)
	var perr *Error
	is.True(errors.As(err, &perr))
	is.Equal(perr.Code, CodeProviderError)
	is.Equal(calls, 3) // initial attempt + 2 retries, then exhausted
}

func TestOpenAIProvider_RateLimitThenSuccess(t *testing.T) {
	is := is.New(t)
	ctrl := gomock.NewController(t)
	svc := mock.NewHTTPService(ctrl)
	withHTTPService(t, svc)

	calls := 0
	svc.EXPECT().Do(gomock.Any(), gomock.Any()).DoAndReturn(
		func(context.Context, pprocutils.HTTPRequest) (pprocutils.HTTPResponse, error) {
			calls++
			if calls == 1 {
				return pprocutils.HTTPResponse{
					StatusCode: http.StatusTooManyRequests,
					Headers:    map[string][]string{"Retry-After": {"0"}},
				}, nil
			}
			return pprocutils.HTTPResponse{
				StatusCode: http.StatusOK,
				Body:       openAISuccessBody(t, map[int]int{0: 2}, 5),
			}, nil
		}).Times(2)

	p, err := newOpenAIProvider(testOpenAIConfig())
	is.NoErr(err)
	result, err := p.Embed(context.Background(), []string{"a"})
	is.NoErr(err)
	is.Equal(len(result.Outcomes), 1)
	is.Equal(calls, 2)
}

func TestOpenAIProvider_MismatchedResponseCount_IsAnError(t *testing.T) {
	is := is.New(t)
	ctrl := gomock.NewController(t)
	svc := mock.NewHTTPService(ctrl)
	withHTTPService(t, svc)

	svc.EXPECT().Do(gomock.Any(), gomock.Any()).Return(pprocutils.HTTPResponse{
		StatusCode: http.StatusOK,
		Body:       openAISuccessBody(t, map[int]int{0: 3}, 1), // only 1 embedding for 2 inputs
	}, nil)

	p, err := newOpenAIProvider(testOpenAIConfig())
	is.NoErr(err)
	_, err = p.Embed(context.Background(), []string{"a", "b"})
	is.True(err != nil) // never silently accept a short/garbled response
}

func TestNewOpenAIProvider_RequiresAuthSecretRef(t *testing.T) {
	is := is.New(t)
	cfg := testOpenAIConfig()
	cfg.OpenAIAuthSecretRef = ""
	_, err := newOpenAIProvider(cfg)
	is.True(err != nil)
	var perr *Error
	is.True(errors.As(err, &perr))
	is.Equal(perr.ConfigPath, "openai.authSecretRef")
}

func TestNewOpenAIProvider_RequiresModel(t *testing.T) {
	is := is.New(t)
	cfg := testOpenAIConfig()
	cfg.Model = ""
	_, err := newOpenAIProvider(cfg)
	is.True(err != nil)
	var perr *Error
	is.True(errors.As(err, &perr))
	is.Equal(perr.ConfigPath, "model")
}
