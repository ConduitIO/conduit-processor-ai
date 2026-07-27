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
	"encoding/json"
	"fmt"
	"strings"
)

// Coded error identifiers. These match the illustrative set in
// docs/design-documents/20260724-ai-pipeline-components.md §9 (Observability)
// in ConduitIO/conduit, plus one Slice-1-specific addition
// (CodeProviderNotImplemented) flagged in the package doc and this repo's
// README — it is not in the design doc's list because the design doc did
// not anticipate providers being built incrementally across slices.
const (
	// CodeNoProviderConfigured is raised when zero embedding-provider
	// candidates resolve (design doc §9: ai.no_provider_configured).
	CodeNoProviderConfigured = "ai.no_provider_configured"
	// CodeAmbiguousProvider is raised when more than one candidate
	// resolves with no explicit selection (design doc §9:
	// ai.ambiguous_provider_configuration).
	CodeAmbiguousProvider = "ai.ambiguous_provider_configuration"
	// CodeProviderNotImplemented is raised when a resolved provider is
	// named and seamed (config, resolution, ambiguity checking) but not yet
	// built. Every provider the design doc §2 names is now implemented, so
	// this is presently unreachable for a valid name; it is retained as a
	// forward-guard for the incremental-slice workflow (see
	// implementedProviders in provider.go) — a future named-but-unbuilt
	// provider fails with this coded error rather than a nil-provider panic.
	// Not part of the design doc's illustrative error-code set.
	CodeProviderNotImplemented = "ai.embedding_provider_not_implemented"
	// CodeProviderError is raised when the provider call itself fails:
	// network, timeout, exhausted rate-limit retries, auth, or a
	// malformed/unexpected response shape (design doc §9:
	// ai.embedding_provider_error). Every batch failure and every
	// per-record failure inside a partially-successful batch is reported
	// under this code.
	CodeProviderError = "ai.embedding_provider_error"
	// CodeInvalidConfig is raised for a structurally invalid processor
	// configuration (failed parsing/validation). Not itself named in the
	// design doc's illustrative set, which focuses on provider-resolution
	// and provider-call failures; this covers the remaining
	// "errors are API" surface every processor needs for its own config.
	CodeInvalidConfig = "ai.invalid_config"
	// CodeFieldResolution is raised when a record's configured
	// input/output field reference cannot be resolved against that
	// record's shape. Per-record, not batch-wide: one record failing
	// field resolution does not fail its sub-batch's other records.
	CodeFieldResolution = "ai.embedding_field_resolution_error"
)

// Error is this processor's stable, --json-serializable error shape:
// every user-facing failure carries a stable Code, the ConfigPath that
// caused it (when the failure is config-shaped, empty otherwise), a
// Suggestion for how to fix it, and the underlying Cause. It implements
// error's Unwrap so callers can still errors.Is/As through to Cause.
type Error struct {
	Code       string `json:"code"`
	Message    string `json:"message"`
	ConfigPath string `json:"configPath,omitempty"`
	Suggestion string `json:"suggestion,omitempty"`
	Cause      error  `json:"-"`
}

func (e *Error) Error() string {
	var b strings.Builder
	fmt.Fprintf(&b, "%s: %s", e.Code, e.Message)
	if e.ConfigPath != "" {
		fmt.Fprintf(&b, " (config path: %s)", e.ConfigPath)
	}
	if e.Suggestion != "" {
		fmt.Fprintf(&b, " — %s", e.Suggestion)
	}
	if e.Cause != nil {
		fmt.Fprintf(&b, ": %v", e.Cause)
	}
	return b.String()
}

func (e *Error) Unwrap() error { return e.Cause }

// MarshalJSON gives Error a stable wire shape independent of the unexported
// Cause field's own JSON-ability — Cause is surfaced as a plain string in
// "cause" instead, so an error whose Cause doesn't implement
// json.Marshaler still serializes cleanly under --json.
func (e *Error) MarshalJSON() ([]byte, error) {
	type alias struct {
		Code       string `json:"code"`
		Message    string `json:"message"`
		ConfigPath string `json:"configPath,omitempty"`
		Suggestion string `json:"suggestion,omitempty"`
		Cause      string `json:"cause,omitempty"`
	}
	a := alias{
		Code:       e.Code,
		Message:    e.Message,
		ConfigPath: e.ConfigPath,
		Suggestion: e.Suggestion,
	}
	if e.Cause != nil {
		a.Cause = e.Cause.Error()
	}
	//nolint:wrapcheck // internal marshal, wrapping would be noise
	return json.Marshal(a)
}

// Is enables errors.Is(err, &Error{Code: "..."}) style code comparisons —
// two *Error values are equal for errors.Is purposes iff their Code
// matches, independent of Message/ConfigPath/Cause.
func (e *Error) Is(target error) bool {
	t, ok := target.(*Error)
	if !ok {
		return false
	}
	return e.Code == t.Code
}

func errNoProviderConfigured() error {
	return &Error{
		Code:    CodeNoProviderConfigured,
		Message: "no embedding provider could be resolved",
		Suggestion: "set the \"provider\" config key (openai, voyage, cohere, or ollama), " +
			"the CONDUIT_EMBED_PROVIDER environment variable, or configure exactly one provider's " +
			"auth secret reference (e.g. openai.authSecretRef) so it can be auto-detected",
	}
}

func errAmbiguousProvider(candidates []string) error {
	return &Error{
		Code: CodeAmbiguousProvider,
		Message: fmt.Sprintf("more than one embedding provider is configured (%s) and none was explicitly selected",
			strings.Join(candidates, ", ")),
		Suggestion: "set the \"provider\" config key or the CONDUIT_EMBED_PROVIDER environment variable " +
			"to pick one explicitly",
	}
}

// errProviderNotImplemented is the forward-guard error for a provider name
// that is whitelisted but has no builder yet (see implementedProviders). It
// is presently unreachable for a valid name — all four named providers are
// built — but is retained so a future named-but-unbuilt provider fails fast
// with a coded error instead of nil-panicking at Open.
func errProviderNotImplemented(name string) error {
	return &Error{
		Code:       CodeProviderNotImplemented,
		Message:    fmt.Sprintf("provider %q is named and seamed but not yet built", name),
		ConfigPath: ConfigProvider,
		Suggestion: "use one of the implemented providers (openai, ollama, voyage, cohere), " +
			"or check the README for when this provider lands",
	}
}

// errUnknownProvider is raised for a provider name that isn't one of the
// four the design doc names at all (a typo, not merely "not implemented
// yet"). Uses CodeInvalidConfig — the same code Config.Validate uses for
// an explicit bad name — so the error is consistent regardless of whether
// the bad name was caught early (explicit "provider" config, at Configure)
// or late (CONDUIT_EMBED_PROVIDER, only resolvable at Open).
func errUnknownProvider(name string) error {
	return &Error{
		Code:       CodeInvalidConfig,
		Message:    fmt.Sprintf("unknown embedding provider %q", name),
		ConfigPath: ConfigProvider,
		Suggestion: "use one of: openai, voyage, cohere, ollama",
	}
}

func newProviderError(provider string, cause error) error {
	return &Error{
		Code:    CodeProviderError,
		Message: fmt.Sprintf("%s embedding call failed", provider),
		Suggestion: "check the referenced auth secret and configured model; if this persists, " +
			"check the provider's status page",
		Cause: cause,
	}
}

func newConfigError(cause error) error {
	return &Error{
		Code:    CodeInvalidConfig,
		Message: "invalid embedding processor configuration",
		Cause:   cause,
	}
}

func newFieldError(configPath string, cause error) error {
	return &Error{
		Code:       CodeFieldResolution,
		Message:    "failed to resolve configured field on this record",
		ConfigPath: configPath,
		Suggestion: "verify the field reference matches this record's shape",
		Cause:      cause,
	}
}
