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

import "context"

// Provider identifiers, shared by config, resolution, metadata, and error
// messages.
const (
	ProviderOpenAI = "openai"
	ProviderVoyage = "voyage"
	ProviderCohere = "cohere"
	ProviderOllama = "ollama"
)

// EnvProvider is the environment variable used for explicit provider
// selection (resolution step 2), mirroring conduit generate's
// CONDUIT_GENERATE_PROVIDER (design doc §2 / 20260722-conduit-generate.md
// Decision §1).
const EnvProvider = "CONDUIT_EMBED_PROVIDER"

// implementedProviders is the set of provider names with a working
// [Provider] adapter. All four the design doc §2 names are now built
// (openai, ollama, voyage, cohere).
//
// The map + [IsImplemented] + [errProviderNotImplemented] mechanism is
// retained deliberately as a forward-guard: it lets a future provider
// constant be named and seamed (config field, resolution, ambiguity
// detection) in one change and built in a later one, failing fast with a
// coded ai.embedding_provider_not_implemented error in the interim rather
// than nil-panicking. With every currently-named provider implemented, the
// guard is presently vacuous — but keeping it costs nothing and preserves
// the incremental-slice workflow the package was built with.
var implementedProviders = map[string]bool{
	ProviderOpenAI: true,
	ProviderOllama: true,
	ProviderVoyage: true,
	ProviderCohere: true,
}

// IsImplemented reports whether name has a working [Provider]. Used for
// fail-fast config validation (Config.Validate) and by tests; see
// implementedProviders for why the mechanism is retained now that all four
// named providers are built.
func IsImplemented(name string) bool {
	return implementedProviders[name]
}

// TokensScope describes what population a [BatchResult]'s TokensUsed
// count covers. Providers report usage at different granularities; this
// package never estimates a finer-grained number than the provider itself
// reports (design doc §2: "never estimated").
type TokensScope int

const (
	// TokensScopeUnknown means the provider did not report usage for this
	// call; TokensUsed is 0 and should not be attached to output metadata.
	TokensScopeUnknown TokensScope = iota
	// TokensScopeRecord means TokensUsed is exact for a single-record
	// call — safe to attach to that one record's metadata as its own
	// cost.
	TokensScopeRecord
	// TokensScopeBatch means TokensUsed is the provider's aggregate count
	// across every input in this call, not divided per record. Attached
	// to every record in the sub-batch verbatim, tagged as batch-scoped
	// (see Processor.attachEmbedding in processor.go) so a consumer never
	// mistakes it for that one record's own cost or double-counts it by
	// summing across records in the same batch.
	TokensScopeBatch
)

// Outcome is one input's result within a [BatchResult]. Exactly
// one of Vector or Err is set.
type Outcome struct {
	Vector []float32
	Err    error
}

// BatchResult is the result of one [Provider.Embed] call. Outcomes has
// exactly len(inputs) elements, in the same order as the inputs passed to
// Embed — [Processor.processBatch] relies on this 1:1 correspondence to
// honor the design doc §4 partial-batch-failure contract without widening
// or dropping any result.
type BatchResult struct {
	Outcomes    []Outcome
	Model       string
	Dimension   int
	TokensUsed  int
	TokensScope TokensScope
}

// Provider embeds a batch of texts against one vendor's API in a single
// host-mediated HTTP call (via the egress package). Implementations must
// perform exactly one egress call per Embed invocation — sub-batching
// across multiple Embed calls within one Process call is [Processor]'s job,
// not the Provider's (see doc.go).
type Provider interface {
	// Name is the provider's stable identifier, used in resolution,
	// output metadata, and error messages. Must equal one of the
	// Provider* constants.
	Name() string
	// MaxBatchSize is the maximum number of inputs this provider accepts
	// in a single Embed call. Processor clamps its own configured
	// maxTextsPerBatch to this value.
	MaxBatchSize() int
	// Embed performs exactly one host-mediated call embedding every
	// element of inputs.
	//
	// A non-nil error means the entire call failed: the caller must treat
	// every input as unembedded (design doc §4, invariants 1/3) — no
	// input in inputs was written to any output.
	//
	// A nil error's BatchResult.Outcomes may still contain per-input
	// errors (Outcome.Err) for a provider response that reports some
	// inputs succeeded and others failed within the same call. The caller
	// honors this 1:1, never widening success to the whole batch and
	// never silently dropping a failed input.
	Embed(ctx context.Context, inputs []string) (BatchResult, error)
}

// ResolveProviderName applies the design doc §2 resolution order:
//  1. Explicit: cfg.Provider.
//  2. Explicit via environment: EnvProvider (CONDUIT_EMBED_PROVIDER).
//  3. Auto-detect exactly one candidate; refuse on zero or ambiguous.
//
// Auto-detection is judged on config-level signals only — whether a
// provider's auth-secret-reference (or, for Ollama, base URL) config field
// is set — never on credential values or environment-variable presence for
// a provider's own API key. Credentials are host-injected per the egress
// package's design and never enter guest memory, so unlike conduit
// generate's provider resolution (which can check e.g. OPENAI_API_KEY
// directly because it runs with a real environment and a real socket),
// this processor cannot inspect "is a hosted API key set" — only "did the
// pipeline author configure a reference to one." See config.go's
// OpenAIAuthSecretRef doc comment.
func ResolveProviderName(cfg Config, getenv func(string) string) (string, error) {
	if cfg.Provider != "" {
		return cfg.Provider, nil
	}
	if getenv != nil {
		if v := getenv(EnvProvider); v != "" {
			return v, nil
		}
	}

	candidates := autoDetectCandidates(cfg)
	switch len(candidates) {
	case 0:
		return "", errNoProviderConfigured()
	case 1:
		return candidates[0], nil
	default:
		return "", errAmbiguousProvider(candidates)
	}
}

// autoDetectCandidates returns providers with a configured credential
// reference (or, for Ollama, a configured base URL), in a fixed order for
// reporting purposes only — order does not imply preference, per the design
// doc §2.
func autoDetectCandidates(cfg Config) []string {
	var candidates []string
	if cfg.OpenAIAuthSecretRef != "" {
		candidates = append(candidates, ProviderOpenAI)
	}
	if cfg.VoyageAuthSecretRef != "" {
		candidates = append(candidates, ProviderVoyage)
	}
	if cfg.CohereAuthSecretRef != "" {
		candidates = append(candidates, ProviderCohere)
	}
	if cfg.OllamaBaseURL != "" {
		candidates = append(candidates, ProviderOllama)
	}
	return candidates
}

// BuildProvider constructs the [Provider] for the resolved provider name.
// Returns a coded ai.invalid_config error for a name that is not one of the
// four the design doc §2 names (a typo, via errUnknownProvider).
func BuildProvider(name string, cfg Config) (Provider, error) {
	switch name {
	case ProviderOpenAI:
		return newOpenAIProvider(cfg)
	case ProviderOllama:
		return newOllamaProvider(cfg)
	case ProviderVoyage:
		return newVoyageProvider(cfg)
	case ProviderCohere:
		return newCohereProvider(cfg)
	default:
		return nil, errUnknownProvider(name)
	}
}
