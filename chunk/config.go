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

import "fmt"

//go:generate go run github.com/conduitio/conduit-commons/paramgen -output=paramgen.go Config

// Strategy identifiers, shared by config, dispatch, and error messages. See
// doc.go for what each one does.
const (
	StrategyFixedSize = "fixed_size"
	StrategySentence  = "sentence"
	StrategyRecursive = "recursive"
)

// Config is the chunking processor's configuration. Field names double as
// the pipeline-config keys (via the paramgen-generated Parameters(), see
// paramgen.go) and every error raised while parsing or validating it names
// the offending key, per CLAUDE.md's "errors are API" convention.
type Config struct {
	// Strategy selects the chunking algorithm: "fixed_size", "sentence", or
	// "recursive". See the package doc for what each one does.
	Strategy string `json:"strategy" default:"fixed_size" validate:"inclusion=fixed_size|sentence|recursive"`

	// ChunkSize is the target maximum chunk size in Unicode characters
	// (runes) — never bytes, see doc.go's "offsets are runes, not bytes"
	// note. fixed_size chunks are exactly ChunkSize runes, except the
	// final one; sentence/recursive chunks are packed up to ChunkSize
	// runes without splitting below their strategy's natural boundary.
	// Default 1000 is an implementation-time constant (design doc §3
	// leaves exact defaults unspecified) chosen as a size that comfortably
	// holds a few paragraphs of prose — small enough for typical embedding
	// model context limits, large enough to avoid over-fragmenting short
	// documents.
	ChunkSize int `json:"chunkSize" default:"1000" validate:"gt=0"`

	// Overlap is the number of trailing runes from one chunk repeated at
	// the start of the next. Only meaningful for, and only honored by, the
	// fixed_size strategy — see Config.Validate. sentence and recursive
	// chunk on natural boundaries (sentences, paragraphs, words) and never
	// repeat content between chunks; a nonzero Overlap with those
	// strategies is accepted but ignored, not an error, so switching
	// strategies doesn't require also clearing this field. Default 100
	// mirrors a common ~10% overlap for the default 1000-rune ChunkSize.
	Overlap int `json:"overlap" default:"100" validate:"gt=-1"`

	// InputField is the record field read as the text to chunk.
	InputField string `json:"inputField" default:".Payload.After"`
	// OutputField is the record field each chunk's text is written to on
	// its output record. Default ".Payload.After.text" writes the chunk's
	// text under a NAMED "text" field of a StructuredData payload — the
	// composable RAG record shape (design doc / RAG contract): a chunk
	// record's After is opencdc.StructuredData{"text": <chunk text>}, never
	// raw bytes, so the sibling embed package's default inputField
	// (".Payload.After.text") reads it directly, and so the embed
	// processor's own default outputField (".Payload.After.vector") can add
	// the vector alongside this text field without clobbering it. Set this
	// explicitly (e.g. back to ".Payload.After") if a pipeline needs the
	// chunk text somewhere else.
	OutputField string `json:"outputField" default:".Payload.After.text"`
}

// Validate checks cross-field invariants paramgen's per-field validations
// (inclusion/gt/...) can't express: Overlap must be strictly less than
// ChunkSize whenever it's actually honored (fixed_size), otherwise the
// sliding window in span.go's fixedSizeSpans never advances and would loop
// forever.
func (c Config) Validate() error {
	if c.Strategy == StrategyFixedSize && c.Overlap >= c.ChunkSize {
		return &Error{
			Code:       CodeInvalidConfig,
			Message:    fmt.Sprintf("overlap (%d) must be less than chunkSize (%d) for the fixed_size strategy", c.Overlap, c.ChunkSize),
			ConfigPath: ConfigOverlap,
			Suggestion: "set overlap to a value smaller than chunkSize",
		}
	}
	return nil
}
