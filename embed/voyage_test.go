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

func testVoyageConfig() Config {
	return Config{
		VoyageAuthSecretRef: "voyage-key",
		VoyageBaseURL:       "https://api.voyageai.com",
		VoyageInputType:     "document",
		VoyageOutputDtype:   "float",
		Model:               "voyage-3.5",
		RequestTimeout:      time.Second,
		MaxRetries:          2,
		RetryBackoffMin:     time.Millisecond,
		RetryBackoffMax:     5 * time.Millisecond,
		RetryBackoffFactor:  2,
	}
}

func voyageSuccessBody(t *testing.T, dims map[int]int, totalTokens int) []byte {
	t.Helper()
	type datum struct {
		Index     int       `json:"index"`
		Embedding []float32 `json:"embedding"`
	}
	resp := struct {
		Object string  `json:"object"`
		Data   []datum `json:"data"`
		Model  string  `json:"model"`
		Usage  struct {
			TotalTokens int `json:"total_tokens"`
		} `json:"usage"`
	}{Object: "list", Model: "voyage-3.5"}
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

func TestVoyageProvider_EmbedSuccess_RequestResponseRoundTrip(t *testing.T) {
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
				Body:       voyageSuccessBody(t, map[int]int{0: 3, 1: 3}, 42),
			}, nil
		})

	p, err := newVoyageProvider(testVoyageConfig())
	is.NoErr(err)

	result, err := p.Embed(context.Background(), []string{"hello", "world"})
	is.NoErr(err)

	// Request round-trip: method, URL, auth secret ref (never a raw
	// Authorization header — that's the host's job), and the marshaled
	// input array plus voyage-specific input_type/output_dtype.
	is.Equal(captured.Method, http.MethodPost)
	is.Equal(captured.URL, "https://api.voyageai.com/v1/embeddings")
	is.Equal(captured.AuthSecretRef, "voyage-key")
	_, hasAuth := captured.Headers["Authorization"]
	is.True(!hasAuth) // credential is host-injected via AuthSecretRef, never a guest header
	var reqBody voyageEmbeddingRequest
	is.NoErr(json.Unmarshal(captured.Body, &reqBody))
	is.Equal(reqBody.Input, []string{"hello", "world"})
	is.Equal(reqBody.Model, "voyage-3.5")
	is.Equal(reqBody.InputType, "document")
	is.Equal(reqBody.OutputDtype, "float")

	// Response round-trip: 1:1 outcomes in input order, dimension, batch-
	// scoped tokens.
	is.Equal(len(result.Outcomes), 2)
	is.Equal(len(result.Outcomes[0].Vector), 3)
	is.Equal(len(result.Outcomes[1].Vector), 3)
	is.NoErr(result.Outcomes[0].Err)
	is.Equal(result.Dimension, 3)
	is.Equal(result.Model, "voyage-3.5")
	is.Equal(result.TokensUsed, 42)
	is.Equal(result.TokensScope, TokensScopeBatch)
}

func TestVoyageProvider_SingleInputTokensScopeIsRecord(t *testing.T) {
	is := is.New(t)
	ctrl := gomock.NewController(t)
	svc := mock.NewHTTPService(ctrl)
	withHTTPService(t, svc)

	svc.EXPECT().Do(gomock.Any(), gomock.Any()).Return(pprocutils.HTTPResponse{
		StatusCode: http.StatusOK,
		Body:       voyageSuccessBody(t, map[int]int{0: 3}, 7),
	}, nil)

	p, err := newVoyageProvider(testVoyageConfig())
	is.NoErr(err)
	result, err := p.Embed(context.Background(), []string{"solo"})
	is.NoErr(err)
	is.Equal(result.TokensScope, TokensScopeRecord)
	is.Equal(result.TokensUsed, 7)
}

func TestVoyageProvider_EmptyInputType_OmitsField(t *testing.T) {
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
				Body:       voyageSuccessBody(t, map[int]int{0: 2}, 3),
			}, nil
		})

	cfg := testVoyageConfig()
	cfg.VoyageInputType = ""
	p, err := newVoyageProvider(cfg)
	is.NoErr(err)
	_, err = p.Embed(context.Background(), []string{"a"})
	is.NoErr(err)

	// An empty input_type must be omitted from the request body entirely,
	// not sent as "" (which Voyage would reject).
	var raw map[string]any
	is.NoErr(json.Unmarshal(captured.Body, &raw))
	_, present := raw["input_type"]
	is.True(!present)
}

func TestVoyageProvider_FullBatchFailure_AuthError_NotRetried(t *testing.T) {
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
				Body:       []byte(`{"detail":"Provided API key is invalid."}`),
			}, nil
		}).Times(1)

	p, err := newVoyageProvider(testVoyageConfig())
	is.NoErr(err)
	_, err = p.Embed(context.Background(), []string{"a", "b"})
	is.True(err != nil)
	var perr *Error
	is.True(errors.As(err, &perr))
	is.Equal(perr.Code, CodeProviderError)
	is.Equal(calls, 1) // 401 must not be retried
}

func TestVoyageProvider_BadRequest_NotRetried(t *testing.T) {
	is := is.New(t)
	ctrl := gomock.NewController(t)
	svc := mock.NewHTTPService(ctrl)
	withHTTPService(t, svc)

	calls := 0
	svc.EXPECT().Do(gomock.Any(), gomock.Any()).DoAndReturn(
		func(context.Context, pprocutils.HTTPRequest) (pprocutils.HTTPResponse, error) {
			calls++
			return pprocutils.HTTPResponse{
				StatusCode: http.StatusBadRequest,
				Body:       []byte(`{"detail":"input is too long"}`),
			}, nil
		}).Times(1)

	p, err := newVoyageProvider(testVoyageConfig())
	is.NoErr(err)
	_, err = p.Embed(context.Background(), []string{"a"})
	is.True(err != nil)
	var perr *Error
	is.True(errors.As(err, &perr))
	is.Equal(perr.Code, CodeProviderError)
	is.Equal(calls, 1) // 400 must not be retried
}

func TestVoyageProvider_RateLimitExhaustion_ReturnsProviderError(t *testing.T) {
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

	cfg := testVoyageConfig()
	cfg.MaxRetries = 2
	p, err := newVoyageProvider(cfg)
	is.NoErr(err)

	_, err = p.Embed(context.Background(), []string{"a"})
	is.True(err != nil)
	var perr *Error
	is.True(errors.As(err, &perr))
	is.Equal(perr.Code, CodeProviderError)
	is.Equal(calls, 3) // initial attempt + 2 retries, then exhausted
}

func TestVoyageProvider_RateLimitThenSuccess(t *testing.T) {
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
				Body:       voyageSuccessBody(t, map[int]int{0: 2}, 5),
			}, nil
		}).Times(2)

	p, err := newVoyageProvider(testVoyageConfig())
	is.NoErr(err)
	result, err := p.Embed(context.Background(), []string{"a"})
	is.NoErr(err)
	is.Equal(len(result.Outcomes), 1)
	is.Equal(calls, 2)
}

func TestVoyageProvider_MismatchedResponseCount_IsAnError(t *testing.T) {
	is := is.New(t)
	ctrl := gomock.NewController(t)
	svc := mock.NewHTTPService(ctrl)
	withHTTPService(t, svc)

	svc.EXPECT().Do(gomock.Any(), gomock.Any()).Return(pprocutils.HTTPResponse{
		StatusCode: http.StatusOK,
		Body:       voyageSuccessBody(t, map[int]int{0: 3}, 1), // only 1 embedding for 2 inputs
	}, nil)

	p, err := newVoyageProvider(testVoyageConfig())
	is.NoErr(err)
	_, err = p.Embed(context.Background(), []string{"a", "b"})
	is.True(err != nil) // never silently accept a short/garbled response
}

func TestVoyageProvider_EmptyInputs_NoCall(t *testing.T) {
	is := is.New(t)
	ctrl := gomock.NewController(t)
	svc := mock.NewHTTPService(ctrl)
	withHTTPService(t, svc)
	// No EXPECT: an empty batch must not reach egress at all.

	p, err := newVoyageProvider(testVoyageConfig())
	is.NoErr(err)
	result, err := p.Embed(context.Background(), nil)
	is.NoErr(err)
	is.Equal(len(result.Outcomes), 0)
}

func TestVoyageProvider_MaxBatchSize(t *testing.T) {
	is := is.New(t)
	p, err := newVoyageProvider(testVoyageConfig())
	is.NoErr(err)
	is.Equal(p.MaxBatchSize(), voyageMaxBatchSize)
	is.Equal(p.MaxBatchSize(), 1000)
}

func TestVoyageProvider_DefaultBaseURLWhenEmpty(t *testing.T) {
	is := is.New(t)
	ctrl := gomock.NewController(t)
	svc := mock.NewHTTPService(ctrl)
	withHTTPService(t, svc)

	var captured pprocutils.HTTPRequest
	svc.EXPECT().Do(gomock.Any(), gomock.Any()).DoAndReturn(
		func(_ context.Context, req pprocutils.HTTPRequest) (pprocutils.HTTPResponse, error) {
			captured = req
			return pprocutils.HTTPResponse{StatusCode: http.StatusOK, Body: voyageSuccessBody(t, map[int]int{0: 2}, 1)}, nil
		})

	cfg := testVoyageConfig()
	cfg.VoyageBaseURL = ""
	p, err := newVoyageProvider(cfg)
	is.NoErr(err)
	_, err = p.Embed(context.Background(), []string{"a"})
	is.NoErr(err)
	is.Equal(captured.URL, defaultVoyageBaseURL+voyageEmbeddingsPath)
}

func TestVoyageProvider_EgressForbidden_CarveOutSuggestion(t *testing.T) {
	is := is.New(t)
	ctrl := gomock.NewController(t)
	svc := mock.NewHTTPService(ctrl)
	withHTTPService(t, svc)

	svc.EXPECT().Do(gomock.Any(), gomock.Any()).Return(
		pprocutils.HTTPResponse{}, egress.ErrForbidden).Times(1)

	p, err := newVoyageProvider(testVoyageConfig())
	is.NoErr(err)
	_, err = p.Embed(context.Background(), []string{"a"})
	is.True(err != nil)
	var perr *Error
	is.True(errors.As(err, &perr))
	is.Equal(perr.Code, CodeProviderError)
	is.True(perr.Suggestion != "")
}

func TestNewVoyageProvider_RequiresAuthSecretRef(t *testing.T) {
	is := is.New(t)
	cfg := testVoyageConfig()
	cfg.VoyageAuthSecretRef = ""
	_, err := newVoyageProvider(cfg)
	is.True(err != nil)
	var perr *Error
	is.True(errors.As(err, &perr))
	is.Equal(perr.ConfigPath, "voyage.authSecretRef")
	is.Equal(perr.Code, CodeInvalidConfig)
}

func TestNewVoyageProvider_RequiresModel(t *testing.T) {
	is := is.New(t)
	cfg := testVoyageConfig()
	cfg.Model = ""
	_, err := newVoyageProvider(cfg)
	is.True(err != nil)
	var perr *Error
	is.True(errors.As(err, &perr))
	is.Equal(perr.ConfigPath, "model")
	is.Equal(perr.Code, CodeInvalidConfig)
}
