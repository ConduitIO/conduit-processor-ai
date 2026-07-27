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

// ollamaMaxBatchSize reflects Ollama's /api/embeddings shape today: one
// "prompt" per request, not an array (design doc §2's provider table:
// "Ollama's /api/embeddings today takes a single input"). This is not a
// self-imposed Config.MaxTextsPerBatch-style ceiling — it is the vendor
// API's actual per-call input count — so Processor.batchSize (processor.go)
// clamps down to it and the sub-batcher ends up issuing one egress call per
// record for this provider. That is the correct, documented behavior for
// Ollama, not a bug or a missed batching optimization: there is no larger
// batch shape this provider's endpoint accepts.
const ollamaMaxBatchSize = 1

const ollamaEmbeddingsPath = "/api/embeddings"

// defaultOllamaBaseURL is used when OllamaBaseURL is unset but "ollama" was
// still explicitly selected (Config.Provider = "ollama" or
// CONDUIT_EMBED_PROVIDER=ollama) rather than auto-detected — auto-detection
// itself still requires OllamaBaseURL to be set (see
// autoDetectCandidates in provider.go), since a bare "Ollama might be
// running on localhost" is too weak a signal to guess at, unlike an
// explicit selection where the pipeline author has already committed to
// this provider and localhost:11434 is Ollama's own documented default.
const defaultOllamaBaseURL = "http://localhost:11434"

// ollamaProvider implements [Provider] by hand-rolling Ollama's
// /api/embeddings request/response JSON and calling [egress.Do] for
// transport — the same shape as [openAIProvider] (see doc.go), and the same
// hand-rolled-JSON precedent the core engine's built-in ollama processor
// already uses for /api/generate
// (pkg/plugin/processor/builtin/impl/ollama/ollama.go in ConduitIO/conduit)
// — no vendored Ollama client exists to reuse, and none is needed for one
// JSON-in/JSON-out endpoint.
//
// Unlike [openAIProvider], ollamaProvider carries no AuthSecretRef: Ollama
// is a local, zero-API-key server (design doc §2), reached through the
// egress package's (IP,port) carve-out for a private/loopback target, not
// a hosted vendor requiring a host-injected Authorization header.
type ollamaProvider struct {
	baseURL string
	model   string
	timeout time.Duration
	retry   retryConfig
}

func newOllamaProvider(cfg Config) (Provider, error) {
	if cfg.Model == "" {
		return nil, &Error{
			Code:       CodeInvalidConfig,
			Message:    "ollama provider selected but model is not configured",
			ConfigPath: ConfigModel,
			Suggestion: `set model to a model pulled on the target ollama server, e.g. "nomic-embed-text"`,
		}
	}

	baseURL := strings.TrimSuffix(cfg.OllamaBaseURL, "/")
	if baseURL == "" {
		baseURL = defaultOllamaBaseURL
	}

	return &ollamaProvider{
		baseURL: baseURL,
		model:   cfg.Model,
		timeout: cfg.RequestTimeout,
		retry: retryConfig{
			Min:        cfg.RetryBackoffMin,
			Max:        cfg.RetryBackoffMax,
			Factor:     cfg.RetryBackoffFactor,
			MaxRetries: cfg.MaxRetries,
		},
	}, nil
}

func (p *ollamaProvider) Name() string { return ProviderOllama }

func (p *ollamaProvider) MaxBatchSize() int { return ollamaMaxBatchSize }

// ollamaEmbeddingRequest is Ollama's POST /api/embeddings request body:
// one "prompt" string, not an array — see ollamaMaxBatchSize.
type ollamaEmbeddingRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
}

// ollamaEmbeddingResponse is Ollama's POST /api/embeddings success response
// body. Unlike OpenAI's response, it carries neither a model echo nor token
// usage — see parseOllamaEmbeddingResponse for how BatchResult.Model and
// TokensScope are derived (or, for tokens, deliberately left unset) from
// that absence.
type ollamaEmbeddingResponse struct {
	Embedding []float32 `json:"embedding"`
}

// ollamaErrorResponse is Ollama's error response body shape.
type ollamaErrorResponse struct {
	Error string `json:"error"`
}

// Embed performs exactly one host-mediated call for exactly one input,
// honoring the [Provider] interface's "exactly one egress call per Embed
// invocation" contract for a provider whose own API only accepts one input
// per call. len(inputs) > 1 never reaches this provider through
// [Processor.processBatch] — [Processor.batchSize] clamps to
// [ollamaProvider.MaxBatchSize] (1) before any sub-batch is formed — but
// Embed still rejects it explicitly rather than silently embedding only
// inputs[0] and dropping the rest, in case a future caller bypasses that
// clamp.
func (p *ollamaProvider) Embed(ctx context.Context, inputs []string) (BatchResult, error) {
	if len(inputs) == 0 {
		return BatchResult{}, nil
	}
	if len(inputs) > 1 {
		return BatchResult{}, fmt.Errorf(
			"ollama provider accepts exactly one input per Embed call, got %d", len(inputs))
	}

	resp, err := p.doEmbedRequest(ctx, inputs[0])
	if err != nil {
		return BatchResult{}, err
	}
	return parseOllamaEmbeddingResponse(resp.Body, p.model)
}

// doEmbedRequest marshals the request, performs the host-mediated call
// (with retry), and returns the raw response — or a coded
// ai.embedding_provider_error if the call or the response status failed.
// Split out of Embed to keep both halves under the linter's function-
// length ceiling, mirroring openai.go's doEmbedRequest.
func (p *ollamaProvider) doEmbedRequest(ctx context.Context, input string) (egress.Response, error) {
	body, err := json.Marshal(ollamaEmbeddingRequest{Model: p.model, Prompt: input})
	if err != nil {
		return egress.Response{}, fmt.Errorf("marshal ollama embeddings request: %w", err)
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
			URL:    p.baseURL + ollamaEmbeddingsPath,
			Headers: map[string][]string{
				"Content-Type": {"application/json"},
			},
			Body: body,
			// No AuthSecretRef: Ollama takes no API key (see the
			// ollamaProvider doc comment above).
		})
	})
	if err != nil {
		return egress.Response{}, classifyOllamaEgressError(err)
	}
	if resp.StatusCode != http.StatusOK {
		return egress.Response{}, classifyOllamaStatus(resp)
	}
	return resp, nil
}

// parseOllamaEmbeddingResponse decodes an Ollama /api/embeddings success
// body into a BatchResult for the single input that produced it.
//
// Model is set to the model the request was made with (model), not
// something echoed back by Ollama's response (it doesn't echo one) — this
// is the literal, known-actual value the call used, not an invented one.
//
// TokensScope is always [TokensScopeUnknown] and TokensUsed is left at its
// zero value: Ollama's /api/embeddings response reports no usage figure at
// all, and this package never estimates a token count a provider didn't
// report (design doc §2: "never estimated"; see attachEmbedding in
// processor.go for how TokensScopeUnknown suppresses the tokensUsed
// metadata entirely rather than attaching a fabricated 0).
func parseOllamaEmbeddingResponse(body []byte, model string) (BatchResult, error) {
	var parsed ollamaEmbeddingResponse
	if err := json.Unmarshal(body, &parsed); err != nil {
		return BatchResult{}, fmt.Errorf("decode ollama embeddings response: %w", err)
	}
	if len(parsed.Embedding) == 0 {
		return BatchResult{}, errors.New("ollama response has an empty or missing embedding")
	}

	return BatchResult{
		Outcomes:    []Outcome{{Vector: parsed.Embedding}},
		Model:       model,
		Dimension:   len(parsed.Embedding),
		TokensScope: TokensScopeUnknown,
	}, nil
}

// classifyOllamaEgressError wraps an egress.Do failure as a coded
// ai.embedding_provider_error, with suggestions tailored to a local server
// reached via the egress package's (IP,port) carve-out rather than a
// hosted vendor.
func classifyOllamaEgressError(err error) error {
	suggestion := "verify the local ollama server is running and reachable at the configured " +
		"ollama.baseURL, and check the pipeline's egress allowlist"
	switch {
	case errors.Is(err, egress.ErrEgressDisabled):
		suggestion = "this processor was not opted into network egress by its operator; " +
			"the pipeline config must allowlist the local ollama server's (IP,port)"
	case errors.Is(err, egress.ErrForbidden):
		suggestion = "the ollama server's (IP,port) is not in the pipeline's egress allowlist; " +
			"a loopback/private target requires an explicit (IP,port) carve-out entry, not a bare hostname"
	case errors.Is(err, egress.ErrTimeout):
		suggestion = "the ollama call exceeded requestTimeout; consider raising it or checking the local server's load"
	}
	return &Error{
		Code:       CodeProviderError,
		Message:    "ollama embeddings call failed",
		Suggestion: suggestion,
		Cause:      err,
	}
}

// classifyOllamaStatus wraps a non-200 Ollama response as a coded
// ai.embedding_provider_error, extracting Ollama's own error message when
// the body parses as JSON. A 429 reaching here means retries were already
// exhausted by doWithRetry, same as openai.go's classifyOpenAIStatus.
func classifyOllamaStatus(resp egress.Response) error {
	var parsed ollamaErrorResponse
	msg := fmt.Sprintf("ollama returned HTTP %d", resp.StatusCode)
	if err := json.Unmarshal(resp.Body, &parsed); err == nil && parsed.Error != "" {
		msg = fmt.Sprintf("ollama returned HTTP %d: %s", resp.StatusCode, parsed.Error)
	}

	suggestion := "check that the local ollama server is running and the configured model is available"
	switch resp.StatusCode {
	case http.StatusNotFound:
		suggestion = "the configured model was not found on the ollama server; pull it first, e.g. `ollama pull <model>`"
	case http.StatusTooManyRequests:
		suggestion = "rate-limit retries were exhausted (maxRetries); consider raising it or reducing request concurrency"
	case http.StatusBadRequest:
		suggestion = "ollama rejected the request; verify the configured model name"
	}

	return &Error{
		Code:       CodeProviderError,
		Message:    msg,
		Suggestion: suggestion,
	}
}
