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

// Package chunk implements Conduit's standalone (WebAssembly) document
// chunking processor: it turns one input record into N output records, one
// per chunk, for the write-path RAG pipeline (Postgres CDC → chunk → embed →
// pgvector) described in
// docs/design-documents/20260724-ai-pipeline-components.md §3 (this
// processor) and §6 (the upsert-idempotency contract this package's chunk_id
// output satisfies) in ConduitIO/conduit.
//
// Unlike the sibling [embed] package, this processor needs no host
// capability at all — it is pure, in-memory text processing
// (Process(records) -> records, no network, no egress). See the design doc's
// §3: "no new host capability needed — pure in-memory text processing."
//
// # Offsets are runes, not bytes
//
// Every offset/length this package reports (chunk config's chunkSize, the
// ai.chunk.offset/ai.chunk.length output metadata) is a count of Unicode
// code points (Go runes), never raw bytes. Slicing a UTF-8 string at a byte
// offset that falls inside a multi-byte rune corrupts it; slicing at a rune
// offset never can, by construction. Every strategy in this package
// converts its input to []rune exactly once (in [Processor.Process]) and
// only ever slices that rune slice, so a chunk boundary can never land
// inside a multi-byte character. See span_test.go's unicode test for the
// property this guarantees.
//
// # Chunks are always contiguous substrings of the source document
//
// Every emitted chunk's text is exactly runes[offset:offset+length] of the
// original input — never a reconstruction, reflow, or join of
// non-adjacent pieces. This is what makes the offset/length metadata
// trustworthy for tracing a chunk back to its exact position in the source
// document, and it is what the round-trip property tests in
// processor_test.go (TestProcessor_RoundTrip_*) check: concatenating a
// record's chunks in index order, with fixed_size's configured overlap
// trimmed back out, reconstructs the original text byte-for-byte.
//
// # Strategies
//
// See span.go for the implementation of all three; config.go's Config.
// Strategy field selects one:
//
//   - fixed_size: a sliding rune-count window of ChunkSize runes, advancing
//     by ChunkSize-Overlap runes each step so the trailing Overlap runes of
//     one chunk repeat at the start of the next. The only strategy that
//     honors Overlap — sentence and recursive chunk on natural boundaries
//     and never repeat content between chunks (see config.go's Overlap doc).
//   - sentence: splits on sentence-ending punctuation ('.', '!', '?')
//     followed by whitespace or end-of-text — a dependency-free heuristic,
//     not an NLP sentence tokenizer, so it has no abbreviation handling
//     ("Dr. Smith" ends a sentence here). Sentences are then packed
//     greedily into chunks up to ChunkSize runes without ever splitting a
//     sentence — a single sentence longer than ChunkSize becomes its own
//     oversized chunk rather than being cut mid-sentence.
//   - recursive: the common "recursive character splitter" shape. Tries
//     paragraph ("\n\n") boundaries first; any resulting piece still over
//     ChunkSize is recursively split on sentence boundaries, then on word
//     (whitespace) boundaries, then — only if a single word still exceeds
//     ChunkSize — hard-split at the rune level. Unlike sentence, recursive
//     guarantees every chunk fits ChunkSize. The finest-grained pieces are
//     then packed greedily back up to ChunkSize, same as sentence.
//
// # Fan-out and chunk_id determinism (design doc §6)
//
// [Processor.Process] returns an [sdk.MultiRecord] for each input record:
// zero elements for empty input text (equivalent to a filter — nothing to
// embed), or N elements, one per chunk. Every chunk's chunk_id is
// `{source_record_key}:{chunk_index}` (0-based) — deterministic from the
// source record's Key and the chunk's position, never random or
// time-based. Two runs over an identical source record produce identical
// chunk_ids, byte for byte; see TestProcessor_DeterministicChunkIDs in
// processor_test.go, the test that proves the property design doc §6's
// pgvector upsert idempotency depends on. A source record with no Key
// cannot produce a stable chunk_id at all — rather than fabricate one, this
// package fails that record with a coded ai.chunk_missing_source_key error
// (errors.go) so a misconfigured upstream is visible, not silently
// corrupting downstream idempotency.
//
// # Tombstones are never chunked
//
// A delete of the source record ([opencdc.OperationDelete]) is recognized
// before any chunking is attempted and is never split into chunk records.
// Instead this package re-emits the same record, unchunked, tagged with an
// ai.chunk.source_key metadata entry naming the deleted row — see
// processor.go's isTombstone and buildDeleteIntent, and the package
// README's "Tombstones / delete-intents" section for the exact shape the
// vector destination consumes. A single source row can have produced many
// chunk rows over its lifetime (chunk count changes as the source document
// is edited), so the delete-intent deliberately does not enumerate
// chunk_ids — the destination resolves "every chunk_id ever derived from
// this source_key" itself (design doc §5).
package chunk
