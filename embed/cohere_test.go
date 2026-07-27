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

	"github.com/conduitio/conduit-processor-sdk/pprocutils"
	"github.com/conduitio/conduit-processor-sdk/pprocutils/mock"
	"github.com/matryer/is"
	"go.uber.org/mock/gomock"
)

func testCohereConfig() Config {
	return Config{
		CohereAuthSecretRef: "cohere-key",
		CohereBaseURL:       "https://api.cohere.com",
		CohereInputType:     "search_document",
		Model:               "embed-english-v3.0",
		RequestTimeout:      time.Second,
		MaxRetries:          2,
		RetryBackoffMin:     time.Millisecond,
		RetryBackoffMax:     5 * time.Millisecond,
		RetryBackoffFactor:  2,
	}
}

// cohereSuccessBody builds the OBJECT-form response (embedding_types set),
// with count embeddings of dimension dim, positionally ordered (no index
// field — this is Cohere's shape). inputTokens < 0 omits billed_units
// entirely; >= 0 includes it.
func cohereSuccessBody(t *testing.T, count, dim, inputTokens int) []byte {
	t.Helper()
	floats := make([][]float32, count)
	for i := range floats {
		vec := make([]float32, dim)
		for j := range vec {
			vec[j] = float32(i*dim+j) * 0.01
		}
		floats[i] = vec
	}
	resp := map[string]any{
		"id":         "abc-123",
		"embeddings": map[string]any{"float": floats},
	}
	if inputTokens >= 0 {
		resp["meta"] = map[string]any{
			"billed_units": map[string]any{"input_tokens": inputTokens},
		}
	}
	body, err := json.Marshal(resp)
	if err != nil {
		t.Fatalf("marshal fixture: %v", err)
	}
	return body
}

func TestCohereProvider_EmbedSuccess_RequestResponseRoundTrip(t *testing.T) {
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
				Body:       cohereSuccessBody(t, 2, 4, 11),
			}, nil
		})

	p, err := newCohereProvider(testCohereConfig())
	is.NoErr(err)

	result, err := p.Embed(context.Background(), []string{"hello", "world"})
	is.NoErr(err)

	// Request round-trip: method, URL, auth secret ref (never a raw
	// Authorization header), the "texts" array, and the divergent Cohere
	// fields: input_type, pinned embedding_types:["float"], truncate.
	is.Equal(captured.Method, http.MethodPost)
	is.Equal(captured.URL, "https://api.cohere.com/v1/embed")
	is.Equal(captured.AuthSecretRef, "cohere-key")
	_, hasAuth := captured.Headers["Authorization"]
	is.True(!hasAuth) // credential is host-injected via AuthSecretRef
	var reqBody cohereEmbedRequest
	is.NoErr(json.Unmarshal(captured.Body, &reqBody))
	is.Equal(reqBody.Texts, []string{"hello", "world"})
	is.Equal(reqBody.Model, "embed-english-v3.0")
	is.Equal(reqBody.InputType, "search_document")
	is.Equal(reqBody.EmbeddingTypes, []string{"float"})
	is.Equal(reqBody.Truncate, "END")

	// Response round-trip: positional 1:1 outcomes, dimension, model set
	// from config (Cohere doesn't echo it), batch-scoped tokens from
	// billed_units.
	is.Equal(len(result.Outcomes), 2)
	is.Equal(len(result.Outcomes[0].Vector), 4)
	is.Equal(len(result.Outcomes[1].Vector), 4)
	is.NoErr(result.Outcomes[0].Err)
	is.Equal(result.Dimension, 4)
	is.Equal(result.Model, "embed-english-v3.0") // set from config, not echoed
	is.Equal(result.TokensUsed, 11)
	is.Equal(result.TokensScope, TokensScopeBatch)
}

// TestCohereProvider_PositionalOrderPreserved proves slot j maps to input j
// (Cohere has no per-item index; order is the only guarantee).
func TestCohereProvider_PositionalOrderPreserved(t *testing.T) {
	is := is.New(t)
	ctrl := gomock.NewController(t)
	svc := mock.NewHTTPService(ctrl)
	withHTTPService(t, svc)

	// Three distinct 1-dim vectors so we can assert order is preserved.
	body := []byte(`{"id":"x","embeddings":{"float":[[0.1],[0.2],[0.3]]},` +
		`"meta":{"billed_units":{"input_tokens":9}}}`)
	svc.EXPECT().Do(gomock.Any(), gomock.Any()).Return(pprocutils.HTTPResponse{
		StatusCode: http.StatusOK, Body: body,
	}, nil)

	p, err := newCohereProvider(testCohereConfig())
	is.NoErr(err)
	result, err := p.Embed(context.Background(), []string{"a", "b", "c"})
	is.NoErr(err)
	is.Equal(len(result.Outcomes), 3)
	is.Equal(result.Outcomes[0].Vector, []float32{0.1})
	is.Equal(result.Outcomes[1].Vector, []float32{0.2})
	is.Equal(result.Outcomes[2].Vector, []float32{0.3})
}

func TestCohereProvider_SingleInputTokensScopeIsRecord(t *testing.T) {
	is := is.New(t)
	ctrl := gomock.NewController(t)
	svc := mock.NewHTTPService(ctrl)
	withHTTPService(t, svc)

	svc.EXPECT().Do(gomock.Any(), gomock.Any()).Return(pprocutils.HTTPResponse{
		StatusCode: http.StatusOK,
		Body:       cohereSuccessBody(t, 1, 3, 5),
	}, nil)

	p, err := newCohereProvider(testCohereConfig())
	is.NoErr(err)
	result, err := p.Embed(context.Background(), []string{"solo"})
	is.NoErr(err)
	is.Equal(result.TokensScope, TokensScopeRecord)
	is.Equal(result.TokensUsed, 5)
}

// TestCohereProvider_AbsentBilledUnits_TokensScopeUnknown covers D-8: a
// missing billed_units must yield TokensScopeUnknown, never a fabricated 0.
func TestCohereProvider_AbsentBilledUnits_TokensScopeUnknown(t *testing.T) {
	is := is.New(t)
	ctrl := gomock.NewController(t)
	svc := mock.NewHTTPService(ctrl)
	withHTTPService(t, svc)

	svc.EXPECT().Do(gomock.Any(), gomock.Any()).Return(pprocutils.HTTPResponse{
		StatusCode: http.StatusOK,
		Body:       cohereSuccessBody(t, 2, 3, -1), // no meta/billed_units
	}, nil)

	p, err := newCohereProvider(testCohereConfig())
	is.NoErr(err)
	result, err := p.Embed(context.Background(), []string{"a", "b"})
	is.NoErr(err)
	is.Equal(result.TokensScope, TokensScopeUnknown)
	is.Equal(result.TokensUsed, 0)
}

func TestCohereProvider_ZeroBilledUnits_TokensScopeUnknown(t *testing.T) {
	is := is.New(t)
	ctrl := gomock.NewController(t)
	svc := mock.NewHTTPService(ctrl)
	withHTTPService(t, svc)

	svc.EXPECT().Do(gomock.Any(), gomock.Any()).Return(pprocutils.HTTPResponse{
		StatusCode: http.StatusOK,
		Body:       cohereSuccessBody(t, 1, 3, 0), // billed_units present but input_tokens 0
	}, nil)

	p, err := newCohereProvider(testCohereConfig())
	is.NoErr(err)
	result, err := p.Embed(context.Background(), []string{"a"})
	is.NoErr(err)
	is.Equal(result.TokensScope, TokensScopeUnknown)
}

// TestCohereProvider_MissingFloatArray_IsError covers D-1's defensive case:
// an "embeddings" object with no "float" key must error, not panic.
func TestCohereProvider_MissingFloatArray_IsError(t *testing.T) {
	is := is.New(t)
	ctrl := gomock.NewController(t)
	svc := mock.NewHTTPService(ctrl)
	withHTTPService(t, svc)

	svc.EXPECT().Do(gomock.Any(), gomock.Any()).Return(pprocutils.HTTPResponse{
		StatusCode: http.StatusOK,
		Body:       []byte(`{"id":"x","embeddings":{},"meta":{"billed_units":{"input_tokens":1}}}`),
	}, nil)

	p, err := newCohereProvider(testCohereConfig())
	is.NoErr(err)
	_, err = p.Embed(context.Background(), []string{"a"})
	is.True(err != nil) // absent float array is a shape failure, not a partial success
}

func TestCohereProvider_MismatchedResponseCount_IsAnError(t *testing.T) {
	is := is.New(t)
	ctrl := gomock.NewController(t)
	svc := mock.NewHTTPService(ctrl)
	withHTTPService(t, svc)

	svc.EXPECT().Do(gomock.Any(), gomock.Any()).Return(pprocutils.HTTPResponse{
		StatusCode: http.StatusOK,
		Body:       cohereSuccessBody(t, 1, 3, 4), // 1 embedding for 2 inputs
	}, nil)

	p, err := newCohereProvider(testCohereConfig())
	is.NoErr(err)
	_, err = p.Embed(context.Background(), []string{"a", "b"})
	is.True(err != nil) // never silently accept a short response
}

func TestCohereProvider_FullBatchFailure_AuthError_NotRetried(t *testing.T) {
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
				Body:       []byte(`{"message":"invalid api token"}`),
			}, nil
		}).Times(1)

	p, err := newCohereProvider(testCohereConfig())
	is.NoErr(err)
	_, err = p.Embed(context.Background(), []string{"a", "b"})
	is.True(err != nil)
	var perr *Error
	is.True(errors.As(err, &perr))
	is.Equal(perr.Code, CodeProviderError)
	is.Equal(calls, 1) // 401 must not be retried
}

func TestCohereProvider_BadRequest_InputTypeSuggestion(t *testing.T) {
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
				Body:       []byte(`{"message":"input_type is required for this model"}`),
			}, nil
		}).Times(1)

	p, err := newCohereProvider(testCohereConfig())
	is.NoErr(err)
	_, err = p.Embed(context.Background(), []string{"a"})
	is.True(err != nil)
	var perr *Error
	is.True(errors.As(err, &perr))
	is.Equal(perr.Code, CodeProviderError)
	is.Equal(calls, 1) // 400 must not be retried
}

func TestCohereProvider_RateLimitExhaustion_ReturnsProviderError(t *testing.T) {
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

	cfg := testCohereConfig()
	cfg.MaxRetries = 2
	p, err := newCohereProvider(cfg)
	is.NoErr(err)

	_, err = p.Embed(context.Background(), []string{"a"})
	is.True(err != nil)
	var perr *Error
	is.True(errors.As(err, &perr))
	is.Equal(perr.Code, CodeProviderError)
	is.Equal(calls, 3) // initial + 2 retries, then exhausted
}

func TestCohereProvider_RateLimitThenSuccess(t *testing.T) {
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
				Body:       cohereSuccessBody(t, 1, 2, 3),
			}, nil
		}).Times(2)

	p, err := newCohereProvider(testCohereConfig())
	is.NoErr(err)
	result, err := p.Embed(context.Background(), []string{"a"})
	is.NoErr(err)
	is.Equal(len(result.Outcomes), 1)
	is.Equal(calls, 2)
}

func TestCohereProvider_EmptyInputs_NoCall(t *testing.T) {
	is := is.New(t)
	ctrl := gomock.NewController(t)
	svc := mock.NewHTTPService(ctrl)
	withHTTPService(t, svc)
	// No EXPECT: an empty batch must not reach egress.

	p, err := newCohereProvider(testCohereConfig())
	is.NoErr(err)
	result, err := p.Embed(context.Background(), nil)
	is.NoErr(err)
	is.Equal(len(result.Outcomes), 0)
}

func TestCohereProvider_MaxBatchSize(t *testing.T) {
	is := is.New(t)
	p, err := newCohereProvider(testCohereConfig())
	is.NoErr(err)
	is.Equal(p.MaxBatchSize(), cohereMaxBatchSize)
	is.Equal(p.MaxBatchSize(), 96)
}

func TestCohereProvider_DefaultBaseURLWhenEmpty(t *testing.T) {
	is := is.New(t)
	ctrl := gomock.NewController(t)
	svc := mock.NewHTTPService(ctrl)
	withHTTPService(t, svc)

	var captured pprocutils.HTTPRequest
	svc.EXPECT().Do(gomock.Any(), gomock.Any()).DoAndReturn(
		func(_ context.Context, req pprocutils.HTTPRequest) (pprocutils.HTTPResponse, error) {
			captured = req
			return pprocutils.HTTPResponse{StatusCode: http.StatusOK, Body: cohereSuccessBody(t, 1, 2, 1)}, nil
		})

	cfg := testCohereConfig()
	cfg.CohereBaseURL = ""
	p, err := newCohereProvider(cfg)
	is.NoErr(err)
	_, err = p.Embed(context.Background(), []string{"a"})
	is.NoErr(err)
	is.Equal(captured.URL, defaultCohereBaseURL+cohereEmbedPath)
}

func TestNewCohereProvider_RequiresAuthSecretRef(t *testing.T) {
	is := is.New(t)
	cfg := testCohereConfig()
	cfg.CohereAuthSecretRef = ""
	_, err := newCohereProvider(cfg)
	is.True(err != nil)
	var perr *Error
	is.True(errors.As(err, &perr))
	is.Equal(perr.ConfigPath, "cohere.authSecretRef")
	is.Equal(perr.Code, CodeInvalidConfig)
}

func TestNewCohereProvider_RequiresModel(t *testing.T) {
	is := is.New(t)
	cfg := testCohereConfig()
	cfg.Model = ""
	_, err := newCohereProvider(cfg)
	is.True(err != nil)
	var perr *Error
	is.True(errors.As(err, &perr))
	is.Equal(perr.ConfigPath, "model")
	is.Equal(perr.Code, CodeInvalidConfig)
}

func TestNewCohereProvider_RejectsInvalidInputType(t *testing.T) {
	is := is.New(t)
	cfg := testCohereConfig()
	cfg.CohereInputType = "not-a-type"
	_, err := newCohereProvider(cfg)
	is.True(err != nil)
	var perr *Error
	is.True(errors.As(err, &perr))
	is.Equal(perr.ConfigPath, "cohere.inputType")
	is.Equal(perr.Code, CodeInvalidConfig)
}

func TestNewCohereProvider_DefaultsInputTypeWhenEmpty(t *testing.T) {
	is := is.New(t)
	cfg := testCohereConfig()
	cfg.CohereInputType = ""
	p, err := newCohereProvider(cfg)
	is.NoErr(err)
	cp, ok := p.(*cohereProvider)
	is.True(ok)
	is.Equal(cp.inputType, "search_document") // defaulted, not left empty (Cohere requires it)
}

func TestNewCohereProvider_AcceptsAllEnumInputTypes(t *testing.T) {
	is := is.New(t)
	for _, it := range []string{"search_document", "search_query", "classification", "clustering"} {
		cfg := testCohereConfig()
		cfg.CohereInputType = it
		_, err := newCohereProvider(cfg)
		is.NoErr(err)
	}
}
