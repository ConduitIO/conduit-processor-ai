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

func testOllamaConfig() Config {
	return Config{
		OllamaBaseURL:      "http://127.0.0.1:11434",
		Model:              "nomic-embed-text",
		RequestTimeout:     time.Second,
		MaxRetries:         2,
		RetryBackoffMin:    time.Millisecond,
		RetryBackoffMax:    5 * time.Millisecond,
		RetryBackoffFactor: 2,
	}
}

func ollamaSuccessBody(t *testing.T, dim int) []byte {
	t.Helper()
	vec := make([]float32, dim)
	for i := range vec {
		vec[i] = float32(i) * 0.01
	}
	body, err := json.Marshal(ollamaEmbeddingResponse{Embedding: vec})
	if err != nil {
		t.Fatalf("marshal fixture: %v", err)
	}
	return body
}

func TestOllamaProvider_EmbedSuccess_RequestResponseRoundTrip(t *testing.T) {
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
				Body:       ollamaSuccessBody(t, 4),
			}, nil
		})

	p, err := newOllamaProvider(testOllamaConfig())
	is.NoErr(err)

	result, err := p.Embed(context.Background(), []string{"hello"})
	is.NoErr(err)

	// Request round-trip: method, URL, no AuthSecretRef (Ollama takes no
	// API key), and the marshaled single-prompt body.
	is.Equal(captured.Method, http.MethodPost)
	is.Equal(captured.URL, "http://127.0.0.1:11434/api/embeddings")
	is.Equal(captured.AuthSecretRef, "")
	var reqBody ollamaEmbeddingRequest
	is.NoErr(json.Unmarshal(captured.Body, &reqBody))
	is.Equal(reqBody.Prompt, "hello")
	is.Equal(reqBody.Model, "nomic-embed-text")

	// Response round-trip: one outcome, model echoed from the request (not
	// invented — Ollama's response doesn't carry one), no tokens reported.
	is.Equal(len(result.Outcomes), 1)
	is.Equal(len(result.Outcomes[0].Vector), 4)
	is.NoErr(result.Outcomes[0].Err)
	is.Equal(result.Model, "nomic-embed-text")
	is.Equal(result.Dimension, 4)
	is.Equal(result.TokensUsed, 0)
	is.Equal(result.TokensScope, TokensScopeUnknown)
}

func TestOllamaProvider_DefaultBaseURL(t *testing.T) {
	is := is.New(t)
	ctrl := gomock.NewController(t)
	svc := mock.NewHTTPService(ctrl)
	withHTTPService(t, svc)

	var captured pprocutils.HTTPRequest
	svc.EXPECT().Do(gomock.Any(), gomock.Any()).DoAndReturn(
		func(_ context.Context, req pprocutils.HTTPRequest) (pprocutils.HTTPResponse, error) {
			captured = req
			return pprocutils.HTTPResponse{StatusCode: http.StatusOK, Body: ollamaSuccessBody(t, 1)}, nil
		})

	cfg := testOllamaConfig()
	cfg.OllamaBaseURL = ""
	p, err := newOllamaProvider(cfg)
	is.NoErr(err)

	_, err = p.Embed(context.Background(), []string{"hello"})
	is.NoErr(err)
	is.Equal(captured.URL, "http://localhost:11434/api/embeddings")
}

func TestOllamaProvider_MaxBatchSizeIsOne(t *testing.T) {
	is := is.New(t)
	p, err := newOllamaProvider(testOllamaConfig())
	is.NoErr(err)
	is.Equal(p.MaxBatchSize(), 1)
}

func TestOllamaProvider_MultipleInputsRejected(t *testing.T) {
	is := is.New(t)
	p, err := newOllamaProvider(testOllamaConfig())
	is.NoErr(err)

	_, err = p.Embed(context.Background(), []string{"a", "b"})
	is.True(err != nil) // Ollama's API takes exactly one prompt per call
}

func TestOllamaProvider_EmptyInputsIsNoOp(t *testing.T) {
	is := is.New(t)
	p, err := newOllamaProvider(testOllamaConfig())
	is.NoErr(err)

	result, err := p.Embed(context.Background(), nil)
	is.NoErr(err)
	is.Equal(len(result.Outcomes), 0)
}

func TestOllamaProvider_FullBatchFailure_NotFound_NotRetried(t *testing.T) {
	is := is.New(t)
	ctrl := gomock.NewController(t)
	svc := mock.NewHTTPService(ctrl)
	withHTTPService(t, svc)

	calls := 0
	svc.EXPECT().Do(gomock.Any(), gomock.Any()).DoAndReturn(
		func(context.Context, pprocutils.HTTPRequest) (pprocutils.HTTPResponse, error) {
			calls++
			return pprocutils.HTTPResponse{
				StatusCode: http.StatusNotFound,
				Body:       []byte(`{"error":"model 'nomic-embed-text' not found"}`),
			}, nil
		}).Times(1)

	p, err := newOllamaProvider(testOllamaConfig())
	is.NoErr(err)
	_, err = p.Embed(context.Background(), []string{"a"})
	is.True(err != nil)
	var perr *Error
	is.True(errors.As(err, &perr))
	is.Equal(perr.Code, CodeProviderError)
	is.Equal(calls, 1) // gomock.Times(1) would fail the test if a retry happened
}

func TestOllamaProvider_RateLimitExhaustion_ReturnsProviderError(t *testing.T) {
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

	cfg := testOllamaConfig()
	cfg.MaxRetries = 2
	p, err := newOllamaProvider(cfg)
	is.NoErr(err)

	_, err = p.Embed(context.Background(), []string{"a"})
	is.True(err != nil)
	var perr *Error
	is.True(errors.As(err, &perr))
	is.Equal(perr.Code, CodeProviderError)
	is.Equal(calls, 3) // initial attempt + 2 retries, then exhausted
}

func TestOllamaProvider_EgressForbidden_IsCodedProviderError(t *testing.T) {
	is := is.New(t)
	ctrl := gomock.NewController(t)
	svc := mock.NewHTTPService(ctrl)
	withHTTPService(t, svc)

	svc.EXPECT().Do(gomock.Any(), gomock.Any()).Return(pprocutils.HTTPResponse{}, egress.ErrForbidden)

	p, err := newOllamaProvider(testOllamaConfig())
	is.NoErr(err)
	_, err = p.Embed(context.Background(), []string{"a"})
	is.True(err != nil)
	var perr *Error
	is.True(errors.As(err, &perr))
	is.Equal(perr.Code, CodeProviderError)
	is.True(errors.Is(err, egress.ErrForbidden))
}

func TestOllamaProvider_EmptyEmbedding_IsAnError(t *testing.T) {
	is := is.New(t)
	ctrl := gomock.NewController(t)
	svc := mock.NewHTTPService(ctrl)
	withHTTPService(t, svc)

	svc.EXPECT().Do(gomock.Any(), gomock.Any()).Return(pprocutils.HTTPResponse{
		StatusCode: http.StatusOK,
		Body:       []byte(`{"embedding":[]}`),
	}, nil)

	p, err := newOllamaProvider(testOllamaConfig())
	is.NoErr(err)
	_, err = p.Embed(context.Background(), []string{"a"})
	is.True(err != nil) // never silently accept a missing/empty embedding
}

func TestNewOllamaProvider_RequiresModel(t *testing.T) {
	is := is.New(t)
	cfg := testOllamaConfig()
	cfg.Model = ""
	_, err := newOllamaProvider(cfg)
	is.True(err != nil)
	var perr *Error
	is.True(errors.As(err, &perr))
	is.Equal(perr.ConfigPath, "model")
}

func TestNewOllamaProvider_NoAuthSecretRefRequired(t *testing.T) {
	is := is.New(t)
	// Unlike openai, ollama never requires an auth secret ref — there is
	// no such config field at all, confirming the provider never asks for
	// one.
	p, err := newOllamaProvider(testOllamaConfig())
	is.NoErr(err)
	is.Equal(p.Name(), ProviderOllama)
}
