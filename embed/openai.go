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
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/conduitio/conduit-processor-sdk/egress"
)

// openAIMaxBatchSize is OpenAI's documented per-request "input" array
// limit for the embeddings endpoint at implementation time — a vendor API
// constraint, not a Conduit-benchmarked figure (CLAUDE.md's
// no-unbenchmarked-performance-claims rule does not apply to a vendor's own
// documented ceiling, but this value should be reverified against current
// OpenAI docs before being relied on for capacity planning). In practice
// Config.MaxTextsPerBatch (default 96) governs sub-batch size well below
// this; MaxBatchSize exists as the hard backstop so a misconfigured
// maxTextsPerBatch above the vendor limit still gets clamped, not rejected
// by the provider.
const openAIMaxBatchSize = 2048

const openAIEmbeddingsPath = "/v1/embeddings"

// openAIProvider implements [Provider] by hand-rolling OpenAI's embeddings
// request/response JSON and calling [egress.Do] for transport. See doc.go
// for why this package does not use the vendored go-openai client the
// core engine's built-in openai.embeddings processor uses: go-openai does
// its own net/http dialing, which cannot run inside this processor's WASM
// guest sandbox.
type openAIProvider struct {
	baseURL       string
	authSecretRef string
	model         string
	timeout       time.Duration
	retry         retryConfig
}

func newOpenAIProvider(cfg Config) (Provider, error) {
	if cfg.OpenAIAuthSecretRef == "" {
		return nil, &Error{
			Code:       CodeInvalidConfig,
			Message:    "openai provider selected but openai.authSecretRef is not configured",
			ConfigPath: ConfigOpenaiAuthSecretRef,
			Suggestion: "set openai.authSecretRef to the name of a host-managed secret holding the OpenAI API key",
		}
	}
	if cfg.Model == "" {
		return nil, &Error{
			Code:       CodeInvalidConfig,
			Message:    "openai provider selected but model is not configured",
			ConfigPath: ConfigModel,
			Suggestion: `set model to an OpenAI embeddings model, e.g. "text-embedding-3-small"`,
		}
	}

	return &openAIProvider{
		baseURL:       strings.TrimSuffix(cfg.OpenAIBaseURL, "/"),
		authSecretRef: cfg.OpenAIAuthSecretRef,
		model:         cfg.Model,
		timeout:       cfg.RequestTimeout,
		retry: retryConfig{
			Min:        cfg.RetryBackoffMin,
			Max:        cfg.RetryBackoffMax,
			Factor:     cfg.RetryBackoffFactor,
			MaxRetries: cfg.MaxRetries,
		},
	}, nil
}

func (p *openAIProvider) Name() string { return ProviderOpenAI }

func (p *openAIProvider) MaxBatchSize() int { return openAIMaxBatchSize }

// openAIEmbeddingRequest is OpenAI's POST /v1/embeddings request body —
// hand-rolled per doc.go, not the go-openai client's request type (which
// would pull in a net/http-dialing dependency this WASM guest cannot run).
type openAIEmbeddingRequest struct {
	Input []string `json:"input"`
	Model string   `json:"model"`
}

// openAIEmbeddingResponse is OpenAI's POST /v1/embeddings success response
// body. Its data[] and batch-level usage are the shared OpenAI/Voyage shape
// (see httpjson.go).
type openAIEmbeddingResponse struct {
	Data  []indexedEmbeddingDatum `json:"data"`
	Model string                  `json:"model"`
	Usage indexedEmbeddingUsage   `json:"usage"`
}

// openAIErrorResponse is OpenAI's error response body shape, used to
// extract an actionable message for [Error.Cause] rather than surfacing a
// raw HTTP status.
type openAIErrorResponse struct {
	Error struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Code    string `json:"code"`
	} `json:"error"`
}

func (p *openAIProvider) Embed(ctx context.Context, inputs []string) (BatchResult, error) {
	if len(inputs) == 0 {
		return BatchResult{}, nil
	}

	resp, err := p.doEmbedRequest(ctx, inputs)
	if err != nil {
		return BatchResult{}, err
	}
	return parseEmbeddingResponse(resp.Body, len(inputs))
}

// doEmbedRequest marshals the request, performs the host-mediated call
// (with retry), and returns the raw response — or a coded
// ai.embedding_provider_error if the call or the response status failed.
// Split out of Embed to keep both halves under the linter's function-
// length ceiling; Embed itself stays the single documented entry point.
func (p *openAIProvider) doEmbedRequest(ctx context.Context, inputs []string) (egress.Response, error) {
	body, err := json.Marshal(openAIEmbeddingRequest{Input: inputs, Model: p.model})
	if err != nil {
		return egress.Response{}, fmt.Errorf("marshal openai embeddings request: %w", err)
	}

	callCtx := ctx
	if p.timeout > 0 {
		var cancel context.CancelFunc
		callCtx, cancel = context.WithTimeout(ctx, p.timeout)
		defer cancel()
	}

	resp, err := doWithRetry(callCtx, p.retry, func(ctx context.Context) (egress.Response, error) {
		return egress.Do(ctx, egress.Request{
			Method: http.MethodPost,
			URL:    p.baseURL + openAIEmbeddingsPath,
			Headers: map[string][]string{
				headerContentType: {mimeJSON},
			},
			Body:          body,
			AuthSecretRef: p.authSecretRef,
		})
	})
	if err != nil {
		return egress.Response{}, classifyHostedEgressError("openai embeddings", err)
	}
	if resp.StatusCode != http.StatusOK {
		return egress.Response{}, classifyOpenAIStatus(resp)
	}
	return resp, nil
}

// parseEmbeddingResponse decodes an OpenAI embeddings success body and maps
// it 1:1 back to its inputs via the shared indexed-response parser (see
// parseIndexedEmbeddings in httpjson.go), rejecting any shape that doesn't
// account for exactly wantCount inputs.
func parseEmbeddingResponse(body []byte, wantCount int) (BatchResult, error) {
	var parsed openAIEmbeddingResponse
	if err := json.Unmarshal(body, &parsed); err != nil {
		return BatchResult{}, fmt.Errorf("decode openai embeddings response: %w", err)
	}
	return parseIndexedEmbeddings("openai", parsed.Data, parsed.Model, parsed.Usage.TotalTokens, wantCount)
}

// classifyOpenAIStatus wraps a non-200 OpenAI response as a coded
// ai.embedding_provider_error, extracting the vendor's own error message
// when the body parses as JSON. A 429 reaching here means retries were
// already exhausted by doWithRetry (design doc §2/§7: exhaustion surfaces
// ai.embedding_provider_error, the batch is not silently dropped or
// acked); a 401/403 reaches here immediately, unretried (Failure Modes
// §1: an auth failure fails fast rather than burning the retry budget).
func classifyOpenAIStatus(resp egress.Response) error {
	var parsed openAIErrorResponse
	msg := fmt.Sprintf("openai returned HTTP %d", resp.StatusCode)
	if err := json.Unmarshal(resp.Body, &parsed); err == nil && parsed.Error.Message != "" {
		msg = fmt.Sprintf("openai returned HTTP %d: %s", resp.StatusCode, parsed.Error.Message)
	}

	suggestion := suggestStatusDefault
	switch resp.StatusCode {
	case http.StatusTooManyRequests:
		suggestion = suggestStatus429
	case http.StatusUnauthorized, http.StatusForbidden:
		suggestion = "the API key referenced by openai.authSecretRef was rejected; verify it is valid and not expired"
	case http.StatusBadRequest:
		suggestion = "openai rejected the request; verify the configured model name and input field contents"
	}

	return &Error{
		Code:       CodeProviderError,
		Message:    msg,
		Suggestion: suggestion,
	}
}
