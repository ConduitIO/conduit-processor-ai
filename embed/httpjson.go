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
	"errors"
	"fmt"

	"github.com/conduitio/conduit-processor-sdk/egress"
)

// Shared request-header constants for every provider's JSON-over-egress call.
const (
	headerContentType = "Content-Type"
	mimeJSON          = "application/json"
)

// Shared error suggestions, identical across the hosted (API-key) providers —
// openai, voyage, cohere. Ollama's egress suggestions differ (it is a local,
// keyless server reached via an (IP,port) carve-out) and live in ollama.go.
const (
	suggestEgressConnectivity = "check network connectivity to the provider and " +
		"the pipeline's egress allowlist configuration"
	suggestEgressDisabled = "this processor was not opted into network egress by its operator; " +
		"the pipeline config must allowlist the embedding provider's host"
	suggestEgressForbidden = "the embedding provider's host is not in the pipeline's egress allowlist"
	suggestEgressTimeout   = "the provider call exceeded requestTimeout; consider raising it or checking provider latency"

	suggestStatusDefault = "check the referenced auth secret and configured model"
	suggestStatus429     = "rate-limit retries were exhausted (maxRetries); " +
		"consider raising it or reducing maxTextsPerBatch"
)

// indexedEmbeddingDatum is one element of an OpenAI-/Voyage-style embeddings
// response: an explicit input index plus its vector. Both providers return
// this shape, so their response structs and parsers share it.
type indexedEmbeddingDatum struct {
	Index     int       `json:"index"`
	Embedding []float32 `json:"embedding"`
}

// indexedEmbeddingUsage is the batch-level token usage OpenAI and Voyage both
// report (not broken down per input, hence [TokensScopeBatch]).
type indexedEmbeddingUsage struct {
	TotalTokens int `json:"total_tokens"`
}

// classifyHostedEgressError wraps an egress.Do failure from a hosted (API-key)
// provider as a coded ai.embedding_provider_error. callLabel is the human
// phrase for the failed call, e.g. "openai embeddings" or "cohere embed".
func classifyHostedEgressError(callLabel string, err error) error {
	suggestion := suggestEgressConnectivity
	switch {
	case errors.Is(err, egress.ErrEgressDisabled):
		suggestion = suggestEgressDisabled
	case errors.Is(err, egress.ErrForbidden):
		suggestion = suggestEgressForbidden
	case errors.Is(err, egress.ErrTimeout):
		suggestion = suggestEgressTimeout
	}
	return &Error{
		Code:       CodeProviderError,
		Message:    callLabel + " call failed",
		Suggestion: suggestion,
		Cause:      err,
	}
}

// parseIndexedEmbeddings maps an OpenAI-shaped embeddings response (a data[]
// where each element carries its explicit input index, plus a batch-level
// token count) into a BatchResult. It rejects any shape that doesn't account
// for exactly wantCount inputs — a short, garbled, or out-of-range-index
// response is a provider-shape failure, never a partial success (invariants
// 1/3). provider names the vendor for error messages. Shared by openai.go and
// voyage.go, whose responses are the same shape.
func parseIndexedEmbeddings(
	provider string, data []indexedEmbeddingDatum, model string, totalTokens, wantCount int,
) (BatchResult, error) {
	if len(data) != wantCount {
		return BatchResult{}, fmt.Errorf(
			"%s returned %d embeddings for a batch of %d inputs", provider, len(data), wantCount)
	}

	outcomes := make([]Outcome, wantCount)
	seen := make([]bool, wantCount)
	dimension := 0
	for _, d := range data {
		if d.Index < 0 || d.Index >= len(outcomes) {
			return BatchResult{}, fmt.Errorf(
				"%s returned out-of-range index %d for a batch of %d", provider, d.Index, wantCount)
		}
		outcomes[d.Index] = Outcome{Vector: d.Embedding}
		seen[d.Index] = true
		if dimension == 0 {
			dimension = len(d.Embedding)
		}
	}
	for i, ok := range seen {
		if !ok {
			return BatchResult{}, fmt.Errorf("%s response is missing an embedding for input index %d", provider, i)
		}
	}

	scope := TokensScopeBatch
	if wantCount == 1 {
		scope = TokensScopeRecord
	}

	return BatchResult{
		Outcomes:    outcomes,
		Model:       model,
		Dimension:   dimension,
		TokensUsed:  totalTokens,
		TokensScope: scope,
	}, nil
}
