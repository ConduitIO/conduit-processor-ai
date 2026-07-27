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

package chunk

import (
	"encoding/json"
	"fmt"
	"strings"
)

// Coded error identifiers. The design doc's illustrative error-code table
// (20260724-ai-pipeline-components.md §9) is scoped to the embedding
// processor and vector destination — it does not enumerate chunking-processor
// codes, so these follow the same "ai.<component>_<condition>" shape by
// convention, matching the sibling embed package's codes.go.
const (
	// CodeInvalidConfig is raised for a structurally invalid processor
	// configuration (failed parsing or Config.Validate's cross-field
	// check).
	CodeInvalidConfig = "ai.chunk_invalid_config"
	// CodeFieldResolution is raised when a record's configured
	// input/output field reference cannot be resolved against that
	// record's shape. Per-record: one record failing field resolution
	// does not affect any other record in the same Process call.
	CodeFieldResolution = "ai.chunk_field_resolution_error"
	// CodeMissingSourceKey is raised when a record (or a tombstone) has no
	// Key, so no deterministic chunk_id (or delete-intent source_key) can
	// be derived for it (design doc §6). Per-record, never a fabricated
	// or random fallback key — see doc.go.
	CodeMissingSourceKey = "ai.chunk_missing_source_key"
)

// Error is this processor's stable, --json-serializable error shape,
// matching the sibling embed package's Error type exactly: every
// user-facing failure carries a stable Code, the ConfigPath that caused it
// (when the failure is config-shaped, empty otherwise), a Suggestion for
// how to fix it, and the underlying Cause. It implements error's Unwrap so
// callers can still errors.Is/As through to Cause.
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

func newConfigError(cause error) error {
	return &Error{
		Code:    CodeInvalidConfig,
		Message: "invalid chunking processor configuration",
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

func newMissingSourceKeyError() error {
	return &Error{
		Code:    CodeMissingSourceKey,
		Message: "record has no Key; a deterministic chunk_id cannot be derived from it",
		Suggestion: "ensure the upstream source/processor sets .Key on every record reaching this " +
			"processor — chunk_id and tombstone delete-intents are both derived from it",
	}
}
