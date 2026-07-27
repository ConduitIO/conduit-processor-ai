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

import "time"

//go:generate go run github.com/conduitio/conduit-commons/paramgen -output=paramgen.go Config

// Config is the embedding processor's configuration. Field names double as
// the pipeline-config keys (via the paramgen-generated Parameters(), see
// paramgen.go) and every error raised while parsing or validating it names
// the offending key, per CLAUDE.md's "errors are API" convention.
type Config struct {
	// Provider explicitly selects the embedding provider ("openai",
	// "voyage", "cohere", or "ollama"). If empty, resolution falls back to
	// the CONDUIT_EMBED_PROVIDER environment variable, then to
	// auto-detecting exactly one configured provider — see ResolveProvider
	// and the design doc's §2 resolution order.
	Provider string `json:"provider"`

	// Model is the provider-specific embedding model identifier, e.g.
	// "text-embedding-3-small" for OpenAI. Required once a provider is
	// resolved; validated in Config.Validate rather than here because the
	// required-ness depends on which provider ends up selected.
	Model string `json:"model"`

	// InputField is the record field read as the text to embed.
	InputField string `json:"inputField" default:".Payload.After"`
	// OutputField is the record field the embedding vector is written to.
	// Defaults to overwriting InputField's target, matching the shipped
	// cohere.embed/openai.embeddings built-ins' convention — set this
	// explicitly (e.g. ".Payload.After.embedding") to preserve the
	// original chunk text alongside its embedding in a structured record.
	OutputField string `json:"outputField" default:".Payload.After"`

	// MaxTextsPerBatch bounds how many records' texts are sent in a single
	// host-mediated embedding call. The processor sub-batches strictly
	// within one Process call — never across calls, see doc.go — into as
	// few calls as this ceiling (clamped to the resolved provider's own
	// per-request input limit) allows. Default 96 mirrors the core
	// engine's built-in cohere.embed processor's MaxTextsPerRequest
	// default and the design doc's §2 proposed default.
	MaxTextsPerBatch int `json:"maxTextsPerBatch" default:"96" validate:"gt=0"`

	// RequestTimeout bounds each individual host-mediated HTTP call.
	// Mirrors the design doc §2's proposed 30s per-attempt deadline.
	RequestTimeout time.Duration `json:"requestTimeout" default:"30s"`

	// MaxRetries bounds retry attempts for a rate-limited (429) or 5xx
	// sub-batch call before it is failed with a coded
	// ai.embedding_provider_error (design doc §2, §7, Failure Modes §1).
	// Retries never cross a sub-batch boundary and never widen a failed
	// batch into a partial success.
	MaxRetries int `json:"maxRetries" default:"5" validate:"gt=-1"`
	// RetryBackoffMin is the minimum wait before the first retry, used
	// when the provider response carries no Retry-After header.
	RetryBackoffMin time.Duration `json:"retryBackoff.min" default:"500ms"`
	// RetryBackoffMax caps the exponential backoff wait between retries.
	RetryBackoffMax time.Duration `json:"retryBackoff.max" default:"30s"`
	// RetryBackoffFactor is the exponential backoff multiplier.
	RetryBackoffFactor float64 `json:"retryBackoff.factor" default:"2" validate:"gt=0"`

	// OpenAIAuthSecretRef names the host-managed secret holding the OpenAI
	// API key. This is never the raw key value — per the egress package's
	// design, credentials are host-injected as the Authorization header
	// immediately before dispatch and never enter guest memory. A
	// non-empty value here is also the auto-detection signal for "openai"
	// as a resolution candidate (see ResolveProvider): the guest cannot
	// see whether an OPENAI_API_KEY-shaped environment variable is set
	// (there is no such guest-visible credential), so candidacy is judged
	// on "did the pipeline author configure a secret reference for this
	// provider" instead.
	OpenAIAuthSecretRef string `json:"openai.authSecretRef"`
	// OpenAIBaseURL overrides the OpenAI API base URL. Must resolve within
	// the pipeline's host-enforced egress allowlist or every call fails
	// with ai.embedding_host_not_allowed (host-side, design doc §1).
	OpenAIBaseURL string `json:"openai.baseURL" default:"https://api.openai.com"`

	// VoyageAuthSecretRef is the Voyage AI equivalent of
	// OpenAIAuthSecretRef. Slice 1 wires this into resolution and
	// ambiguity detection but the "voyage" provider itself is not yet
	// implemented — see doc.go's Slice 1 scope note.
	VoyageAuthSecretRef string `json:"voyage.authSecretRef"`
	// CohereAuthSecretRef is the Cohere equivalent of OpenAIAuthSecretRef.
	// Not yet implemented in this slice; see doc.go.
	CohereAuthSecretRef string `json:"cohere.authSecretRef"`
	// OllamaBaseURL, if set, makes "ollama" an auto-detection candidate
	// (local Ollama has no API key) and overrides the ollama provider's
	// default target, "http://localhost:11434" (Ollama's own documented
	// default; used when "ollama" is explicitly selected but this field is
	// left empty). Must resolve within the pipeline's host-enforced egress
	// allowlist — for a loopback/private target that means an explicit
	// (IP,port) carve-out entry, not a bare hostname allowlist entry (see
	// the egress package's design doc).
	OllamaBaseURL string `json:"ollama.baseURL"`
}
