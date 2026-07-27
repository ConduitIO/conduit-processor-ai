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
			ConfigPath: "openai.authSecretRef",
			Suggestion: "set openai.authSecretRef to the name of a host-managed secret holding the OpenAI API key",
		}
	}
	if cfg.Model == "" {
		return nil, &Error{
			Code:       CodeInvalidConfig,
			Message:    "openai provider selected but model is not configured",
			ConfigPath: "model",
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
// body.
type openAIEmbeddingResponse struct {
	Data  []openAIEmbeddingDatum `json:"data"`
	Model string                 `json:"model"`
	Usage openAIUsage            `json:"usage"`
}

type openAIEmbeddingDatum struct {
	Index     int       `json:"index"`
	Embedding []float32 `json:"embedding"`
}

// openAIUsage is OpenAI's batch-level token usage. It is not broken down
// per input, hence [TokensScopeBatch] — see provider.go's TokensScope doc.
type openAIUsage struct {
	TotalTokens int `json:"total_tokens"`
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
				"Content-Type": {"application/json"},
			},
			Body:          body,
			AuthSecretRef: p.authSecretRef,
		})
	})
	if err != nil {
		return egress.Response{}, classifyEgressError(err)
	}
	if resp.StatusCode != http.StatusOK {
		return egress.Response{}, classifyOpenAIStatus(resp)
	}
	return resp, nil
}

// parseEmbeddingResponse decodes an OpenAI embeddings success body into a
// BatchResult, mapping each returned datum back to its input index 1:1 and
// rejecting (rather than silently accepting) any shape that doesn't
// account for exactly wantCount inputs — a short, garbled, or
// out-of-range response is a provider-shape failure, not a partial
// success.
func parseEmbeddingResponse(body []byte, wantCount int) (BatchResult, error) {
	var parsed openAIEmbeddingResponse
	if err := json.Unmarshal(body, &parsed); err != nil {
		return BatchResult{}, fmt.Errorf("decode openai embeddings response: %w", err)
	}
	if len(parsed.Data) != wantCount {
		return BatchResult{}, fmt.Errorf(
			"openai returned %d embeddings for a batch of %d inputs", len(parsed.Data), wantCount)
	}

	outcomes := make([]Outcome, wantCount)
	seen := make([]bool, wantCount)
	dimension := 0
	for _, d := range parsed.Data {
		if d.Index < 0 || d.Index >= len(outcomes) {
			return BatchResult{}, fmt.Errorf(
				"openai returned out-of-range index %d for a batch of %d", d.Index, wantCount)
		}
		outcomes[d.Index] = Outcome{Vector: d.Embedding}
		seen[d.Index] = true
		if dimension == 0 {
			dimension = len(d.Embedding)
		}
	}
	for i, ok := range seen {
		if !ok {
			return BatchResult{}, fmt.Errorf("openai response is missing an embedding for input index %d", i)
		}
	}

	scope := TokensScopeBatch
	if wantCount == 1 {
		scope = TokensScopeRecord
	}

	return BatchResult{
		Outcomes:    outcomes,
		Model:       parsed.Model,
		Dimension:   dimension,
		TokensUsed:  parsed.Usage.TotalTokens,
		TokensScope: scope,
	}, nil
}

// classifyEgressError wraps an egress.Do failure (a transport-level or
// policy-level rejection, or exhausted retries on a transient one) as a
// coded ai.embedding_provider_error.
func classifyEgressError(err error) error {
	suggestion := "check network connectivity to the provider and the pipeline's egress allowlist configuration"
	switch {
	case errors.Is(err, egress.ErrEgressDisabled):
		suggestion = "this processor was not opted into network egress by its operator; " +
			"the pipeline config must allowlist the embedding provider's host"
	case errors.Is(err, egress.ErrForbidden):
		suggestion = "the embedding provider's host is not in the pipeline's egress allowlist"
	case errors.Is(err, egress.ErrTimeout):
		suggestion = "the provider call exceeded requestTimeout; consider raising it or checking provider latency"
	}
	return &Error{
		Code:       CodeProviderError,
		Message:    "openai embeddings call failed",
		Suggestion: suggestion,
		Cause:      err,
	}
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

	suggestion := "check the referenced auth secret and configured model"
	switch resp.StatusCode {
	case http.StatusTooManyRequests:
		suggestion = "rate-limit retries were exhausted (maxRetries); consider raising it or reducing maxTextsPerBatch"
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
