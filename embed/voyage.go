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

// voyageMaxBatchSize is Voyage's documented per-request "input" array limit
// for the embeddings endpoint at implementation time — a vendor API
// constraint, not a Conduit-benchmarked figure (reverify against current
// Voyage docs before relying on it for capacity planning). Voyage also caps
// total tokens per request per model, which this package does not model (it
// does not tokenize in-guest): an oversized batch surfaces as a 400 and is
// classified like any other bad-request failure. In practice
// Config.MaxTextsPerBatch (default 96) governs sub-batch size well below
// this ceiling; MaxBatchSize exists as the hard backstop so a misconfigured
// maxTextsPerBatch above the vendor limit still gets clamped, not rejected.
const voyageMaxBatchSize = 1000

const voyageEmbeddingsPath = "/v1/embeddings"

// defaultVoyageBaseURL is used when VoyageBaseURL is unset (mirrors the
// paramgen default so a directly-constructed provider still targets the
// real host).
const defaultVoyageBaseURL = "https://api.voyageai.com"

// voyageProvider implements [Provider] by hand-rolling Voyage's embeddings
// request/response JSON and calling [egress.Do] for transport — the same
// shape as [openAIProvider] (see doc.go for why this package hand-rolls JSON
// rather than using a vendor SDK: the vendor client dials net/http itself and
// cannot run in this processor's WASM guest sandbox). Voyage's request and
// response are OpenAI-shaped (an input array in, a data[] of index+embedding
// out, with batch-level usage), so this adapter is the lowest-divergence of
// the four.
type voyageProvider struct {
	baseURL       string
	authSecretRef string
	model         string
	inputType     string
	outputDtype   string
	timeout       time.Duration
	retry         retryConfig
}

func newVoyageProvider(cfg Config) (Provider, error) {
	if cfg.VoyageAuthSecretRef == "" {
		return nil, &Error{
			Code:       CodeInvalidConfig,
			Message:    "voyage provider selected but voyage.authSecretRef is not configured",
			ConfigPath: ConfigVoyageAuthSecretRef,
			Suggestion: "set voyage.authSecretRef to the name of a host-managed secret holding the Voyage API key",
		}
	}
	if cfg.Model == "" {
		return nil, &Error{
			Code:       CodeInvalidConfig,
			Message:    "voyage provider selected but model is not configured",
			ConfigPath: ConfigModel,
			Suggestion: `set model to a Voyage embeddings model, e.g. "voyage-3.5"`,
		}
	}

	baseURL := strings.TrimSuffix(cfg.VoyageBaseURL, "/")
	if baseURL == "" {
		baseURL = defaultVoyageBaseURL
	}

	return &voyageProvider{
		baseURL:       baseURL,
		authSecretRef: cfg.VoyageAuthSecretRef,
		model:         cfg.Model,
		inputType:     cfg.VoyageInputType,
		outputDtype:   cfg.VoyageOutputDtype,
		timeout:       cfg.RequestTimeout,
		retry: retryConfig{
			Min:        cfg.RetryBackoffMin,
			Max:        cfg.RetryBackoffMax,
			Factor:     cfg.RetryBackoffFactor,
			MaxRetries: cfg.MaxRetries,
		},
	}, nil
}

func (p *voyageProvider) Name() string { return ProviderVoyage }

func (p *voyageProvider) MaxBatchSize() int { return voyageMaxBatchSize }

// voyageEmbeddingRequest is Voyage's POST /v1/embeddings request body —
// hand-rolled per doc.go. input_type and output_dtype are omitted when empty;
// output_dtype is pinned to "float" by config default because pgvector's
// vector path accepts only float embeddings (int8/binary dtypes are out of
// scope for this slice).
type voyageEmbeddingRequest struct {
	Input       []string `json:"input"`
	Model       string   `json:"model"`
	InputType   string   `json:"input_type,omitempty"`
	OutputDtype string   `json:"output_dtype,omitempty"`
}

// voyageEmbeddingResponse is Voyage's POST /v1/embeddings success response
// body — OpenAI-shaped: a data[] of index+embedding, plus batch-level usage
// (the shared shape in httpjson.go).
type voyageEmbeddingResponse struct {
	Data  []indexedEmbeddingDatum `json:"data"`
	Model string                  `json:"model"`
	Usage indexedEmbeddingUsage   `json:"usage"`
}

// voyageErrorResponse is Voyage's error response body shape. Voyage reports
// errors under "detail"; "error" is accepted as a fallback in case a given
// endpoint/status uses the OpenAI-style field instead.
type voyageErrorResponse struct {
	Detail string `json:"detail"`
	Error  string `json:"error"`
}

func (p *voyageProvider) Embed(ctx context.Context, inputs []string) (BatchResult, error) {
	if len(inputs) == 0 {
		return BatchResult{}, nil
	}

	resp, err := p.doEmbedRequest(ctx, inputs)
	if err != nil {
		return BatchResult{}, err
	}
	return parseVoyageResponse(resp.Body, len(inputs))
}

// doEmbedRequest marshals the request, performs the host-mediated call (with
// retry), and returns the raw response — or a coded
// ai.embedding_provider_error if the call or the response status failed.
// Split out of Embed to keep both halves under the linter's function-length
// ceiling, mirroring openai.go's doEmbedRequest.
func (p *voyageProvider) doEmbedRequest(ctx context.Context, inputs []string) (egress.Response, error) {
	body, err := json.Marshal(voyageEmbeddingRequest{
		Input:       inputs,
		Model:       p.model,
		InputType:   p.inputType,
		OutputDtype: p.outputDtype,
	})
	if err != nil {
		return egress.Response{}, fmt.Errorf("marshal voyage embeddings request: %w", err)
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
			URL:    p.baseURL + voyageEmbeddingsPath,
			Headers: map[string][]string{
				headerContentType: {mimeJSON},
			},
			Body:          body,
			AuthSecretRef: p.authSecretRef,
		})
	})
	if err != nil {
		return egress.Response{}, classifyHostedEgressError("voyage embeddings", err)
	}
	if resp.StatusCode != http.StatusOK {
		return egress.Response{}, classifyVoyageStatus(resp)
	}
	return resp, nil
}

// parseVoyageResponse decodes a Voyage embeddings success body and maps it
// 1:1 back to its inputs via the shared indexed-response parser (see
// parseIndexedEmbeddings in httpjson.go), rejecting any shape that doesn't
// account for exactly wantCount inputs. Voyage's response is OpenAI-shaped.
func parseVoyageResponse(body []byte, wantCount int) (BatchResult, error) {
	var parsed voyageEmbeddingResponse
	if err := json.Unmarshal(body, &parsed); err != nil {
		return BatchResult{}, fmt.Errorf("decode voyage embeddings response: %w", err)
	}
	return parseIndexedEmbeddings("voyage", parsed.Data, parsed.Model, parsed.Usage.TotalTokens, wantCount)
}

// classifyVoyageStatus wraps a non-200 Voyage response as a coded
// ai.embedding_provider_error, extracting Voyage's own error message when the
// body parses as JSON. A 429 reaching here means retries were already
// exhausted by doWithRetry; a 401/403 reaches here immediately, unretried
// (an auth failure fails fast rather than burning the retry budget). Mirrors
// classifyOpenAIStatus (openai.go).
func classifyVoyageStatus(resp egress.Response) error {
	var parsed voyageErrorResponse
	msg := fmt.Sprintf("voyage returned HTTP %d", resp.StatusCode)
	if err := json.Unmarshal(resp.Body, &parsed); err == nil {
		detail := parsed.Detail
		if detail == "" {
			detail = parsed.Error
		}
		if detail != "" {
			msg = fmt.Sprintf("voyage returned HTTP %d: %s", resp.StatusCode, detail)
		}
	}

	suggestion := suggestStatusDefault
	switch resp.StatusCode {
	case http.StatusTooManyRequests:
		suggestion = suggestStatus429
	case http.StatusUnauthorized, http.StatusForbidden:
		suggestion = "the API key referenced by voyage.authSecretRef was rejected; verify it is valid and not expired"
	case http.StatusBadRequest:
		suggestion = "voyage rejected the request; verify the configured model name, input contents, " +
			"and that voyage.outputDtype is a supported value (this slice supports \"float\")"
	}

	return &Error{
		Code:       CodeProviderError,
		Message:    msg,
		Suggestion: suggestion,
	}
}
