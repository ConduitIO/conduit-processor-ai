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
	"context"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"

	"github.com/conduitio/conduit-commons/config"
	"github.com/conduitio/conduit-commons/opencdc"
	"github.com/conduitio/conduit-processor-ai/internal/version"
	sdk "github.com/conduitio/conduit-processor-sdk"
)

// Metadata keys attached to every chunk record this processor emits. Field
// names are stable, part of this processor's output contract (design doc
// §3/§6) — the pgvector destination (and any other vector-destination
// consumer) matches on these exact keys, so they are never renamed without
// a version bump and a migration note (CLAUDE.md's breaking-change
// discipline for public contracts).
const (
	// MetadataChunkID is the deterministic chunk_id
	// ({source_record_key}:{chunk_index}) — design doc §6's upsert
	// idempotency key.
	MetadataChunkID = "ai.chunk.id"
	// MetadataSourceKey is the source record's Key, stringified. Present
	// on every chunk record AND on tombstone delete-intents (doc.go), so
	// a vector destination can delete every chunk_id ever derived from
	// one source row without enumerating them (design doc §5).
	MetadataSourceKey = "ai.chunk.source_key"
	// MetadataChunkIndex is the chunk's 0-based position among its
	// source record's chunks, as a base-10 string.
	MetadataChunkIndex = "ai.chunk.index"
	// MetadataChunkOffset is the chunk's start offset in the source
	// document, in runes (not bytes — doc.go), as a base-10 string.
	MetadataChunkOffset = "ai.chunk.offset"
	// MetadataChunkLength is the chunk's length in runes (not bytes —
	// doc.go), as a base-10 string.
	MetadataChunkLength = "ai.chunk.length"
)

// Processor is the standalone (WebAssembly) chunking processor. It
// implements sdk.Processor's Specification/Configure/Open/Process via
// embedding sdk.UnimplementedProcessor for the methods this slice doesn't
// need to override (Teardown, MiddlewareOptions). Unlike the sibling embed
// package's Processor, this one performs no I/O and needs no host
// capability — see doc.go.
type Processor struct {
	sdk.UnimplementedProcessor

	config         Config
	inputResolver  sdk.ReferenceResolver
	outputResolver sdk.ReferenceResolver
}

// NewProcessor constructs a chunking Processor ready for Configure.
func NewProcessor() *Processor {
	return &Processor{}
}

func (p *Processor) Specification() (sdk.Specification, error) {
	return sdk.Specification{
		Name:    "ai.chunk",
		Summary: "Split a record's text into N chunk records using a configurable strategy.",
		Description: "Reads the configured inputField, splits its text into chunks using strategy " +
			"(fixed_size, sentence, or recursive), and emits one output record per chunk, each carrying a " +
			"deterministic chunk_id and offset/length metadata. A tombstone (delete) of the source record is " +
			"never chunked — it is re-emitted unchanged, tagged with the source record's key, as a delete-intent " +
			"for the vector destination to resolve into per-chunk deletes. See the package README's delivery-" +
			"semantics and metadata-contract sections.",
		Version:    version.Value,
		Author:     "Meroxa, Inc.",
		Parameters: Config{}.Parameters(),
	}, nil
}

func (p *Processor) Configure(ctx context.Context, cfg config.Config) error {
	var parsed Config
	if err := sdk.ParseConfig(ctx, cfg, &parsed, Config{}.Parameters()); err != nil {
		return newConfigError(err)
	}
	if err := parsed.Validate(); err != nil {
		return err
	}
	p.config = parsed
	return nil
}

// Open prepares the field resolvers. It performs no I/O of its own, so it
// does not use its context parameter, but keeps it to satisfy
// sdk.Processor's Open(context.Context) error contract.
func (p *Processor) Open(_ context.Context) error {
	inputResolver, err := sdk.NewReferenceResolver(p.config.InputField)
	if err != nil {
		return fmt.Errorf("failed to create a field resolver for inputField %q: %w", p.config.InputField, err)
	}
	p.inputResolver = inputResolver

	outputResolver, err := sdk.NewReferenceResolver(p.config.OutputField)
	if err != nil {
		return fmt.Errorf("failed to create a field resolver for outputField %q: %w", p.config.OutputField, err)
	}
	p.outputResolver = outputResolver

	return nil
}

// isTombstone reports whether rec represents a delete of the source
// record. This is opencdc's own definition of a delete
// (Operation == OperationDelete, "Payload should be populated for all
// operations except OperationDelete") rather than an inference from an
// empty payload — a record can legitimately have an empty/nil After for
// reasons unrelated to deletion, and this package doesn't want a false
// positive silently swallowing content that should have been chunked.
func isTombstone(rec *opencdc.Record) bool {
	return rec.Operation == opencdc.OperationDelete
}

// Process turns each input record into zero, one, or many output records:
//   - A tombstone (isTombstone) is never chunked — see buildDeleteIntent.
//   - Otherwise the record's resolved input text is split into chunks
//     (splitSpans); zero chunks (empty text) is reported as an
//     sdk.MultiRecord with no elements (equivalent to a filter — nothing
//     to embed downstream, see doc.go), one or more chunks become an
//     sdk.MultiRecord with one element per chunk, in order.
//
// The returned slice always has the same length as records, one
// sdk.ProcessedRecord per input record at the same index — the interface
// contract sdk.Processor.Process documents. Every per-record failure
// (unresolvable field, missing source key) is isolated to that record's
// slot and never affects any other record in the same call.
func (p *Processor) Process(_ context.Context, records []opencdc.Record) []sdk.ProcessedRecord {
	out := make([]sdk.ProcessedRecord, len(records))
	for i := range records {
		out[i] = p.processOne(&records[i])
	}
	return out
}

func (p *Processor) processOne(rec *opencdc.Record) sdk.ProcessedRecord {
	sourceKey, err := p.resolveSourceKey(rec)
	if err != nil {
		return sdk.ErrorRecord{Error: err}
	}

	if isTombstone(rec) {
		deleteIntent := p.buildDeleteIntent(rec, sourceKey)
		return sdk.SingleRecord(deleteIntent)
	}

	text, err := p.resolveInputText(rec)
	if err != nil {
		return sdk.ErrorRecord{Error: newFieldError(p.config.InputField, err)}
	}

	runes := []rune(text)
	spans, err := splitSpans(runes, p.config)
	if err != nil {
		// Unreachable via the normal Configure path (paramgen's inclusion
		// validation already rejects an unknown strategy) — defensive only.
		return sdk.ErrorRecord{Error: newConfigError(err)}
	}
	if len(spans) == 0 {
		// Empty input text: nothing to embed downstream. Equivalent to
		// FilterRecord per sdk.MultiRecord's documented zero-element
		// behavior — see doc.go.
		return sdk.MultiRecord{}
	}

	chunks := make([]opencdc.Record, len(spans))
	for i, sp := range spans {
		chunkRec, err := p.buildChunkRecord(rec, sourceKey, i, sp, runes)
		if err != nil {
			// A field-resolution failure while writing the chunk text is a
			// processor-shape bug (the same outputField resolved fine for
			// every other chunk of this record), not a per-chunk data
			// condition — fail the whole record rather than emit a
			// partially-chunked, silently-incomplete result.
			return sdk.ErrorRecord{Error: err}
		}
		chunks[i] = chunkRec
	}
	return sdk.MultiRecord(chunks)
}

// resolveInputText resolves Config.InputField on rec and coerces it to a
// string, mirroring the sibling embed package's resolveInputText exactly
// (same accepted shapes: opencdc.Position, opencdc.Data, string).
func (p *Processor) resolveInputText(rec *opencdc.Record) (string, error) {
	ref, err := p.inputResolver.Resolve(rec)
	if err != nil {
		return "", fmt.Errorf("resolve inputField: %w", err)
	}
	switch v := ref.Get().(type) {
	case opencdc.Position:
		return string(v), nil
	case opencdc.Data:
		return string(v.Bytes()), nil
	case string:
		return v, nil
	case nil:
		return "", fmt.Errorf("inputField resolved to a nil value")
	default:
		return "", fmt.Errorf("inputField resolved to unsupported type %T", v)
	}
}

// resolveSourceKey stringifies rec.Key into the source_key this package's
// chunk_id and delete-intent metadata are both derived from (design doc
// §6). RawData is used verbatim (already a byte string); StructuredData is
// marshaled with the standard library's encoding/json rather than
// opencdc.Data.Bytes()'s own serializer, specifically because
// encoding/json.Marshal of a map guarantees alphabetically sorted keys —
// the determinism property this package's chunk_id contract depends on
// must not accidentally rely on Go map iteration order or on a
// serializer's internal (and not contractually documented) key ordering.
// A nil or empty Key can't produce a stable id at all and is a coded error
// (errors.go's CodeMissingSourceKey), never a fabricated fallback.
func (p *Processor) resolveSourceKey(rec *opencdc.Record) (string, error) {
	switch v := rec.Key.(type) {
	case nil:
		return "", newMissingSourceKeyError()
	case opencdc.RawData:
		if len(v) == 0 {
			return "", newMissingSourceKeyError()
		}
		return string(v), nil
	case opencdc.StructuredData:
		if len(v) == 0 {
			return "", newMissingSourceKeyError()
		}
		b, err := json.Marshal(map[string]any(v))
		if err != nil {
			return "", fmt.Errorf("marshal structured key: %w", err)
		}
		return string(b), nil
	default:
		return "", fmt.Errorf("record key resolved to unsupported type %T", v)
	}
}

// buildChunkRecord derives one chunk's output record from rec: a deep
// clone (so sibling chunks never share a mutable Metadata map) with the
// chunk's text written to Config.OutputField and the ai.chunk.* metadata
// contract (design doc §3/§6) attached. Key is set to the chunk_id itself
// — the natural per-row identity a destination connector's default
// upsert-by-Key behavior can use directly, in addition to the same value
// being available explicitly as ai.chunk.id metadata for consumers that
// prefer not to depend on Key's reuse. Position and Operation are carried
// over unchanged from the source record (see the package README for why).
func (p *Processor) buildChunkRecord(rec *opencdc.Record, sourceKey string, idx int, sp span, runes []rune) (opencdc.Record, error) {
	out := rec.Clone()

	// If outputField writes into a structured subfield of Payload.After
	// (the default, ".Payload.After.text"), the field resolver's walk only
	// descends into Payload.After when it is already nil or
	// opencdc.StructuredData — it refuses to walk into a non-structured
	// value (config.go's default inputField, ".Payload.After", is commonly
	// RawData or a plain string coming straight off a CDC source). This
	// package's contract always replaces the chunk record's After payload
	// with a fresh StructuredData (never merges chunk text into whatever
	// raw bytes the source carried), so reset it here so the resolver can
	// create the structured field it needs. A Payload.After that is
	// ALREADY StructuredData (e.g. a source row with multiple columns) is
	// left untouched here — its other fields are preserved as siblings of
	// the chunk text, exactly like TestProcessor_RecordShape_StructuredPayload
	// exercises.
	if strings.HasPrefix(p.config.OutputField, ".Payload.After.") {
		switch out.Payload.After.(type) {
		case opencdc.StructuredData, nil:
			// already walkable — leave as is (preserves sibling fields).
		default:
			out.Payload.After = nil
		}
	}

	ref, err := p.outputResolver.Resolve(&out)
	if err != nil {
		return opencdc.Record{}, newFieldError(p.config.OutputField, fmt.Errorf("resolve outputField: %w", err))
	}
	if err := ref.Set(string(runes[sp.start:sp.end])); err != nil {
		return opencdc.Record{}, newFieldError(p.config.OutputField, fmt.Errorf("set outputField: %w", err))
	}

	chunkID := fmt.Sprintf("%s:%d", sourceKey, idx)
	out.Key = opencdc.RawData(chunkID)

	if out.Metadata == nil {
		out.Metadata = opencdc.Metadata{}
	}
	out.Metadata[MetadataChunkID] = chunkID
	out.Metadata[MetadataSourceKey] = sourceKey
	out.Metadata[MetadataChunkIndex] = strconv.Itoa(idx)
	out.Metadata[MetadataChunkOffset] = strconv.Itoa(sp.start)
	out.Metadata[MetadataChunkLength] = strconv.Itoa(sp.len())

	return out, nil
}

// buildDeleteIntent re-emits rec unchanged (same Operation, Key, Payload —
// still a normal opencdc delete any destination understands on its own)
// with one addition: ai.chunk.source_key, so a vector destination that
// understands this processor's contract can resolve "every chunk_id ever
// derived from this source row" and delete them all, without this
// processor enumerating chunk_ids it doesn't have (design doc §5 — chunk
// count can change between edits of the same source row, so the previous
// chunk set isn't necessarily this processor's to know). See doc.go's
// "tombstones are never chunked" section.
func (p *Processor) buildDeleteIntent(rec *opencdc.Record, sourceKey string) opencdc.Record {
	out := rec.Clone()
	if out.Metadata == nil {
		out.Metadata = opencdc.Metadata{}
	}
	out.Metadata[MetadataSourceKey] = sourceKey
	return out
}
