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
	"sort"
	"strings"
	"time"

	"github.com/conduitio/conduit-processor-sdk/egress"
)

// cohereMaxBatchSize is Cohere's documented hard per-call "texts" limit for
// the v1 embed endpoint at implementation time — a vendor API constraint,
// not a Conduit-benchmarked figure. This is exactly why Config.MaxTextsPerBatch
// defaults to 96 (see config.go); so batchSize() clamps to 96 with no
// effective change for the default config, but the clamp is load-bearing if
// an operator raises maxTextsPerBatch above it.
const cohereMaxBatchSize = 96

const cohereEmbedPath = "/v1/embed"

// defaultCohereBaseURL is used when CohereBaseURL is unset (mirrors the
// paramgen default so a directly-constructed provider still targets the real
// host).
const defaultCohereBaseURL = "https://api.cohere.com"

const (
	// cohereInputTypeSearchDocument is the default (and the value for
	// embedding stored RAG chunks) of Cohere's REQUIRED input_type field.
	cohereInputTypeSearchDocument = "search_document"
	// cohereDtypeFloat is the sole embedding_types value this slice requests
	// (pgvector's float vector path; int8/binary are out of scope). Pinning
	// it also forces the response's object form (see cohereEmbedResponse).
	cohereDtypeFloat = "float"
)

// cohereInputTypes is the enum Cohere's REQUIRED input_type field must be one
// of. Validated at provider construction so a bad value fails fast with a
// coded config error, naming cohere.inputType, rather than surfacing later as
// an opaque 400 from the vendor.
var cohereInputTypes = map[string]bool{
	cohereInputTypeSearchDocument: true,
	"search_query":                true,
	"classification":              true,
	"clustering":                  true,
}

// cohereProvider implements [Provider] by hand-rolling Cohere's v1 embed
// request/response JSON and calling [egress.Do] for transport (see doc.go for
// why this package hand-rolls JSON rather than using cohere-go: the vendor
// client dials net/http itself and cannot run in this processor's WASM guest
// sandbox).
//
// Cohere's shape diverges from OpenAI/Voyage in three load-bearing ways, and
// this adapter is the reason the provider seam is not a copy-paste:
//
//   - input_type is REQUIRED for Cohere's v3 embedding models — omitting it
//     is a 400 — so it is a validated, defaulted config key (cohere.inputType,
//     default "search_document" for stored RAG chunks).
//   - embeddings are NOT under a "data" key and carry no per-item index. This
//     adapter pins embedding_types:["float"] in the request so the response's
//     "embeddings" field is the OBJECT form ({"float":[[...]]}) rather than the
//     bare array form Cohere returns when embedding_types is omitted; the
//     embeddings are then read positionally, slot j mapping to input j.
//   - the response does not reliably echo the model, so BatchResult.Model is
//     set to the literal configured model (the known-actual value the call
//     used), exactly as ollama.go does — never an invented one.
type cohereProvider struct {
	baseURL       string
	authSecretRef string
	model         string
	inputType     string
	timeout       time.Duration
	retry         retryConfig
}

func newCohereProvider(cfg Config) (Provider, error) {
	if cfg.CohereAuthSecretRef == "" {
		return nil, &Error{
			Code:       CodeInvalidConfig,
			Message:    "cohere provider selected but cohere.authSecretRef is not configured",
			ConfigPath: ConfigCohereAuthSecretRef,
			Suggestion: "set cohere.authSecretRef to the name of a host-managed secret holding the Cohere API key",
		}
	}
	if cfg.Model == "" {
		return nil, &Error{
			Code:       CodeInvalidConfig,
			Message:    "cohere provider selected but model is not configured",
			ConfigPath: ConfigModel,
			Suggestion: `set model to a Cohere embedding model, e.g. "embed-english-v3.0"`,
		}
	}

	inputType := cfg.CohereInputType
	if inputType == "" {
		inputType = cohereInputTypeSearchDocument
	}
	if !cohereInputTypes[inputType] {
		return nil, &Error{
			Code:       CodeInvalidConfig,
			Message:    fmt.Sprintf("cohere.inputType %q is not a valid Cohere input_type", inputType),
			ConfigPath: ConfigCohereInputType,
			Suggestion: "set cohere.inputType to one of: " + strings.Join(sortedCohereInputTypes(), ", ") +
				" (use \"search_document\" for embedding stored RAG chunks)",
		}
	}

	baseURL := strings.TrimSuffix(cfg.CohereBaseURL, "/")
	if baseURL == "" {
		baseURL = defaultCohereBaseURL
	}

	return &cohereProvider{
		baseURL:       baseURL,
		authSecretRef: cfg.CohereAuthSecretRef,
		model:         cfg.Model,
		inputType:     inputType,
		timeout:       cfg.RequestTimeout,
		retry: retryConfig{
			Min:        cfg.RetryBackoffMin,
			Max:        cfg.RetryBackoffMax,
			Factor:     cfg.RetryBackoffFactor,
			MaxRetries: cfg.MaxRetries,
		},
	}, nil
}

// sortedCohereInputTypes returns the accepted input_type values in a stable
// order for deterministic error messages.
func sortedCohereInputTypes() []string {
	out := make([]string, 0, len(cohereInputTypes))
	for k := range cohereInputTypes {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

func (p *cohereProvider) Name() string { return ProviderCohere }

func (p *cohereProvider) MaxBatchSize() int { return cohereMaxBatchSize }

// cohereEmbedRequest is Cohere's POST /v1/embed request body — hand-rolled
// per doc.go. embedding_types is pinned to ["float"] so the response's
// "embeddings" field is the object form ({"float":[[...]]}), read
// deterministically by parseCohereResponse; truncate "END" is Cohere's own
// default, made explicit here.
type cohereEmbedRequest struct {
	Texts          []string `json:"texts"`
	Model          string   `json:"model"`
	InputType      string   `json:"input_type"`
	EmbeddingTypes []string `json:"embedding_types"`
	Truncate       string   `json:"truncate"`
}

// cohereEmbedResponse is Cohere's POST /v1/embed success response body (with
// embedding_types set, so "embeddings" is the object form). Embeddings are
// positionally aligned with the request's "texts" — there is no per-item
// index field to cross-check against (see parseCohereResponse).
type cohereEmbedResponse struct {
	ID         string           `json:"id"`
	Embeddings cohereEmbeddings `json:"embeddings"`
	Meta       cohereMeta       `json:"meta"`
}

// cohereEmbeddings is the object-form "embeddings" field: one array-of-arrays
// per requested dtype. This adapter requests only "float".
type cohereEmbeddings struct {
	Float [][]float32 `json:"float"`
}

type cohereMeta struct {
	BilledUnits cohereBilledUnits `json:"billed_units"`
}

// cohereBilledUnits carries Cohere's batch-level token usage. input_tokens
// may be absent (zero) on some models/plans; parseCohereResponse treats an
// absent/zero count as TokensScopeUnknown and suppresses the metadata rather
// than emitting a fabricated 0 (the "literal, never invented" rule).
type cohereBilledUnits struct {
	InputTokens int `json:"input_tokens"`
}

// cohereErrorResponse is Cohere's error response body shape (a top-level
// "message").
type cohereErrorResponse struct {
	Message string `json:"message"`
}

func (p *cohereProvider) Embed(ctx context.Context, inputs []string) (BatchResult, error) {
	if len(inputs) == 0 {
		return BatchResult{}, nil
	}

	resp, err := p.doEmbedRequest(ctx, inputs)
	if err != nil {
		return BatchResult{}, err
	}
	return parseCohereResponse(resp.Body, len(inputs), p.model)
}

// doEmbedRequest marshals the request, performs the host-mediated call (with
// retry), and returns the raw response — or a coded
// ai.embedding_provider_error if the call or the response status failed.
// Split out of Embed to keep both halves under the linter's function-length
// ceiling, mirroring openai.go's doEmbedRequest.
func (p *cohereProvider) doEmbedRequest(ctx context.Context, inputs []string) (egress.Response, error) {
	body, err := json.Marshal(cohereEmbedRequest{
		Texts:          inputs,
		Model:          p.model,
		InputType:      p.inputType,
		EmbeddingTypes: []string{cohereDtypeFloat},
		Truncate:       "END",
	})
	if err != nil {
		return egress.Response{}, fmt.Errorf("marshal cohere embed request: %w", err)
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
			URL:    p.baseURL + cohereEmbedPath,
			Headers: map[string][]string{
				headerContentType: {mimeJSON},
			},
			Body:          body,
			AuthSecretRef: p.authSecretRef,
		})
	})
	if err != nil {
		return egress.Response{}, classifyHostedEgressError("cohere embed", err)
	}
	if resp.StatusCode != http.StatusOK {
		return egress.Response{}, classifyCohereStatus(resp)
	}
	return resp, nil
}

// parseCohereResponse decodes a Cohere v1 embed success body into a
// BatchResult. Unlike OpenAI/Voyage, Cohere returns embeddings positionally
// with no index field, so this maps slot j to input j and relies on Cohere's
// documented input-order guarantee; the only structural guard is the exact
// length check (len(embeddings.float) == wantCount) — a short, missing, or
// object-shape-absent "float" array is a provider-shape failure, never a
// partial success. Model is set to the caller's configured model (the
// response does not reliably echo one), matching ollama.go's "literal, never
// invented" rule.
func parseCohereResponse(body []byte, wantCount int, model string) (BatchResult, error) {
	var parsed cohereEmbedResponse
	if err := json.Unmarshal(body, &parsed); err != nil {
		return BatchResult{}, fmt.Errorf("decode cohere embed response: %w", err)
	}

	vecs := parsed.Embeddings.Float
	if len(vecs) != wantCount {
		return BatchResult{}, fmt.Errorf(
			"cohere returned %d embeddings for a batch of %d inputs (embeddings.float)", len(vecs), wantCount)
	}

	outcomes := make([]Outcome, wantCount)
	dimension := 0
	for j, v := range vecs {
		if len(v) == 0 {
			return BatchResult{}, fmt.Errorf("cohere returned an empty embedding for input index %d", j)
		}
		outcomes[j] = Outcome{Vector: v}
		if dimension == 0 {
			dimension = len(v)
		}
	}

	scope := TokensScopeBatch
	if wantCount == 1 {
		scope = TokensScopeRecord
	}
	// Absent/zero billed_units → suppress the metadata rather than emit a
	// fabricated 0 (the "literal, never invented" token rule; design doc §2).
	tokens := parsed.Meta.BilledUnits.InputTokens
	if tokens == 0 {
		scope = TokensScopeUnknown
	}

	return BatchResult{
		Outcomes:    outcomes,
		Model:       model,
		Dimension:   dimension,
		TokensUsed:  tokens,
		TokensScope: scope,
	}, nil
}

// classifyCohereStatus wraps a non-200 Cohere response as a coded
// ai.embedding_provider_error, extracting Cohere's own error message (a
// top-level "message") when the body parses as JSON. A 429 reaching here
// means retries were already exhausted by doWithRetry; a 401/403 reaches here
// immediately, unretried. A 400 gets a suggestion pointing at cohere.inputType
// since a missing/invalid input_type is Cohere's most common bad-request
// cause. Mirrors classifyOpenAIStatus (openai.go).
func classifyCohereStatus(resp egress.Response) error {
	var parsed cohereErrorResponse
	msg := fmt.Sprintf("cohere returned HTTP %d", resp.StatusCode)
	if err := json.Unmarshal(resp.Body, &parsed); err == nil && parsed.Message != "" {
		msg = fmt.Sprintf("cohere returned HTTP %d: %s", resp.StatusCode, parsed.Message)
	}

	suggestion := suggestStatusDefault
	switch resp.StatusCode {
	case http.StatusTooManyRequests:
		suggestion = suggestStatus429
	case http.StatusUnauthorized, http.StatusForbidden:
		suggestion = "the API key referenced by cohere.authSecretRef was rejected; verify it is valid and not expired"
	case http.StatusBadRequest:
		suggestion = "cohere rejected the request; verify the configured model name and that cohere.inputType is set " +
			"to a value the model accepts (v3 models require input_type)"
	}

	return &Error{
		Code:       CodeProviderError,
		Message:    msg,
		Suggestion: suggestion,
	}
}
