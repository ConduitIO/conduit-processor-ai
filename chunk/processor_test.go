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
	"errors"
	"strconv"
	"strings"
	"testing"

	"github.com/conduitio/conduit-commons/config"
	"github.com/conduitio/conduit-commons/opencdc"
	sdk "github.com/conduitio/conduit-processor-sdk"
	"github.com/matryer/is"
)

func newTestProcessor(t *testing.T, cfgOverrides map[string]string) *Processor {
	t.Helper()
	is := is.New(t)

	p := NewProcessor()
	cfg := config.Config{}
	for k, v := range cfgOverrides {
		cfg[k] = v
	}
	is.NoErr(p.Configure(context.Background(), cfg))
	is.NoErr(p.Open(context.Background()))
	return p
}

func recordWithKeyAndText(key, text string) opencdc.Record {
	return opencdc.Record{
		Operation: opencdc.OperationCreate,
		Key:       opencdc.RawData(key),
		Metadata:  opencdc.Metadata{},
		Payload:   opencdc.Change{After: opencdc.RawData(text)},
	}
}

// chunkText extracts a chunk record's text under the default outputField
// contract shape (opencdc.StructuredData{"text": ...}) — the composable RAG
// record shape (design doc / RAG contract): a chunk record's After is never
// raw bytes, so tests default-configured (outputField unset) must read the
// "text" field rather than treat Payload.After.Bytes() as the chunk text
// itself.
func chunkText(t *testing.T, rec opencdc.Record) string {
	t.Helper()
	is := is.New(t)
	after, ok := rec.Payload.After.(opencdc.StructuredData)
	is.True(ok) // default outputField always produces a StructuredData payload
	text, ok := after["text"].(string)
	is.True(ok)
	return text
}

// --- fan-out ---

func TestProcessor_FanOutCount(t *testing.T) {
	is := is.New(t)
	p := newTestProcessor(t, map[string]string{
		"strategy": "fixed_size", "chunkSize": "10", "overlap": "0",
	})
	rec := recordWithKeyAndText("doc-1", strings.Repeat("a", 25)) // -> 3 chunks: 10/10/5

	out := p.Process(context.Background(), []opencdc.Record{rec})

	is.Equal(len(out), 1) // one ProcessedRecord slot per input record
	mr, ok := out[0].(sdk.MultiRecord)
	is.True(ok)
	is.Equal(len(mr), 3) // fanned out into 3 chunk records
}

func TestProcessor_FanOutCountVariesPerRecordInSameCall(t *testing.T) {
	is := is.New(t)
	p := newTestProcessor(t, map[string]string{
		"strategy": "fixed_size", "chunkSize": "10", "overlap": "0",
	})
	records := []opencdc.Record{
		recordWithKeyAndText("doc-1", strings.Repeat("a", 5)),  // 1 chunk
		recordWithKeyAndText("doc-2", strings.Repeat("b", 25)), // 3 chunks
		recordWithKeyAndText("doc-3", strings.Repeat("c", 20)), // 2 chunks
	}

	out := p.Process(context.Background(), records)

	is.Equal(len(out), 3) // 1:1 with input records, regardless of each one's fan-out width
	is.Equal(len(out[0].(sdk.MultiRecord)), 1)
	is.Equal(len(out[1].(sdk.MultiRecord)), 3)
	is.Equal(len(out[2].(sdk.MultiRecord)), 2)
}

func TestProcessor_EmptyText_ZeroChunksNotAnError(t *testing.T) {
	is := is.New(t)
	p := newTestProcessor(t, nil)
	rec := recordWithKeyAndText("doc-1", "")

	out := p.Process(context.Background(), []opencdc.Record{rec})

	is.Equal(len(out), 1)
	mr, ok := out[0].(sdk.MultiRecord)
	is.True(ok)
	is.Equal(len(mr), 0) // filtered, not an error - see doc.go
}

// --- chunk_id determinism (design doc §6 - the idempotency property) ---

func TestProcessor_DeterministicChunkIDs(t *testing.T) {
	is := is.New(t)
	rec := recordWithKeyAndText("order-42", strings.Repeat("The quick brown fox. ", 20))

	run := func() []string {
		// A fresh Processor instance each run, exactly like two independent
		// pipeline restarts (or two retried attempts) would see: nothing
		// in-process is carried over between runs.
		p := newTestProcessor(t, map[string]string{
			"strategy": "fixed_size", "chunkSize": "50", "overlap": "5",
		})
		out := p.Process(context.Background(), []opencdc.Record{rec.Clone()})
		mr := out[0].(sdk.MultiRecord)
		ids := make([]string, len(mr))
		for i, r := range mr {
			ids[i] = r.Metadata[MetadataChunkID]
		}
		return ids
	}

	first := run()
	second := run()

	is.True(len(first) > 1) // exercise fan-out, not a degenerate single-chunk case
	is.Equal(first, second) // byte-for-byte identical across independent runs
	for i, id := range first {
		is.Equal(id, "order-42:"+strconv.Itoa(i)) // and matches the documented {source_key}:{index} shape
	}
}

func TestProcessor_DeterministicChunkIDs_StructuredKey(t *testing.T) {
	is := is.New(t)
	rec := opencdc.Record{
		Operation: opencdc.OperationCreate,
		Key:       opencdc.StructuredData{"id": 42, "shard": "b"},
		Payload:   opencdc.Change{After: opencdc.RawData(strings.Repeat("word ", 30))},
	}

	run := func() string {
		p := newTestProcessor(t, map[string]string{"chunkSize": "20", "overlap": "0"})
		out := p.Process(context.Background(), []opencdc.Record{rec.Clone()})
		mr := out[0].(sdk.MultiRecord)
		is.True(len(mr) > 0)
		return mr[0].Metadata[MetadataChunkID]
	}

	// encoding/json.Marshal of a Go map is documented to sort keys
	// alphabetically, so a StructuredData key's stringification - and
	// therefore the chunk_id derived from it - must be stable across
	// runs regardless of Go's randomized map iteration order.
	is.Equal(run(), run())
}

// --- metadata contract (exact keys the vector destination consumes) ---

func TestProcessor_MetadataContract(t *testing.T) {
	is := is.New(t)
	p := newTestProcessor(t, map[string]string{
		"strategy": "fixed_size", "chunkSize": "10", "overlap": "0",
	})
	text := strings.Repeat("a", 25)
	rec := recordWithKeyAndText("doc-1", text)

	out := p.Process(context.Background(), []opencdc.Record{rec})
	mr := out[0].(sdk.MultiRecord)
	is.Equal(len(mr), 3)

	runes := []rune(text)
	for i, chunkRec := range mr {
		is.Equal(chunkRec.Metadata[MetadataChunkID], "doc-1:"+strconv.Itoa(i))
		is.Equal(chunkRec.Metadata[MetadataSourceKey], "doc-1")
		is.Equal(chunkRec.Metadata[MetadataChunkIndex], strconv.Itoa(i))

		offset, err := strconv.Atoi(chunkRec.Metadata[MetadataChunkOffset])
		is.NoErr(err)
		length, err := strconv.Atoi(chunkRec.Metadata[MetadataChunkLength])
		is.NoErr(err)

		// offset/length round-trip to exactly this chunk's payload text -
		// the metadata isn't just present, it's correct against the
		// original document.
		want := string(runes[offset : offset+length])
		is.Equal(chunkText(t, chunkRec), want)

		// Key carries the same chunk_id, giving a destination connector's
		// default upsert-by-Key behavior the right identity for free.
		is.Equal(string(chunkRec.Key.Bytes()), chunkRec.Metadata[MetadataChunkID])
	}
}

func TestProcessor_OffsetLengthCorrectForSentenceStrategy(t *testing.T) {
	is := is.New(t)
	p := newTestProcessor(t, map[string]string{
		"strategy": "sentence", "chunkSize": "1000",
	})
	text := "First sentence here. Second one follows! Third?"
	rec := recordWithKeyAndText("doc-1", text)

	out := p.Process(context.Background(), []opencdc.Record{rec})
	mr := out[0].(sdk.MultiRecord)
	runes := []rune(text)

	for _, chunkRec := range mr {
		offset, _ := strconv.Atoi(chunkRec.Metadata[MetadataChunkOffset])
		length, _ := strconv.Atoi(chunkRec.Metadata[MetadataChunkLength])
		want := string(runes[offset : offset+length])
		is.Equal(chunkText(t, chunkRec), want)
	}
}

// --- tombstones: never chunked, never dropped ---

func TestProcessor_TombstoneNeverChunkedNeverDropped(t *testing.T) {
	is := is.New(t)
	p := newTestProcessor(t, map[string]string{"chunkSize": "5", "overlap": "0"}) // small enough that Before would fan out heavily if chunked
	rec := opencdc.Record{
		Operation: opencdc.OperationDelete,
		Key:       opencdc.RawData("doc-1"),
		Metadata:  opencdc.Metadata{},
		Payload:   opencdc.Change{Before: opencdc.RawData(strings.Repeat("would fan out into many chunks ", 20))},
	}

	out := p.Process(context.Background(), []opencdc.Record{rec})

	is.Equal(len(out), 1)
	single, ok := out[0].(sdk.SingleRecord)
	is.True(ok) // never a MultiRecord - the whole point of the tombstone path
	got := opencdc.Record(single)
	is.Equal(got.Operation, opencdc.OperationDelete)
	is.Equal(string(got.Key.Bytes()), "doc-1")
	is.Equal(got.Metadata[MetadataSourceKey], "doc-1")
	// The delete-intent carries no chunk_id/index/offset/length - it isn't
	// one chunk, it's a signal to delete every chunk ever derived from
	// this source key (design doc §5).
	_, hasChunkID := got.Metadata[MetadataChunkID]
	is.True(!hasChunkID)
	// Payload is passed through completely unchanged - still a normal
	// opencdc delete any destination understands without this
	// processor's contract.
	is.Equal(got.Payload, rec.Payload)
}

func TestProcessor_TombstoneMissingKey_ErrorsNotSilentlyDropped(t *testing.T) {
	is := is.New(t)
	p := newTestProcessor(t, nil)
	rec := opencdc.Record{
		Operation: opencdc.OperationDelete,
		Metadata:  opencdc.Metadata{},
		Payload:   opencdc.Change{Before: opencdc.RawData("some content")},
	}

	out := p.Process(context.Background(), []opencdc.Record{rec})

	is.Equal(len(out), 1)
	errRec, ok := out[0].(sdk.ErrorRecord)
	is.True(ok) // not silently filtered/dropped - an unresolvable delete-intent is a loud failure
	var chunkErr *Error
	is.True(errors.As(errRec.Error, &chunkErr))
	is.Equal(chunkErr.Code, CodeMissingSourceKey)
}

// --- missing source key on ordinary (non-tombstone) records ---

func TestProcessor_MissingSourceKey_NonTombstone(t *testing.T) {
	is := is.New(t)
	p := newTestProcessor(t, nil)
	rec := opencdc.Record{
		Operation: opencdc.OperationCreate,
		Metadata:  opencdc.Metadata{},
		Payload:   opencdc.Change{After: opencdc.RawData("some content to chunk")},
	}

	out := p.Process(context.Background(), []opencdc.Record{rec})

	errRec, ok := out[0].(sdk.ErrorRecord)
	is.True(ok)
	var chunkErr *Error
	is.True(errors.As(errRec.Error, &chunkErr))
	is.Equal(chunkErr.Code, CodeMissingSourceKey)
}

// --- per-record isolation: one bad record never affects its siblings ---

func TestProcessor_PerRecordFailureIsolated(t *testing.T) {
	is := is.New(t)
	p := newTestProcessor(t, map[string]string{"chunkSize": "1000"})
	good1 := recordWithKeyAndText("doc-1", "some good text here")
	bad := opencdc.Record{Operation: opencdc.OperationCreate, Payload: opencdc.Change{After: opencdc.RawData("no key")}} // missing key
	good2 := recordWithKeyAndText("doc-2", "more good text here")

	out := p.Process(context.Background(), []opencdc.Record{good1, bad, good2})

	is.Equal(len(out), 3)
	_, ok := out[0].(sdk.MultiRecord)
	is.True(ok)
	_, ok = out[1].(sdk.ErrorRecord)
	is.True(ok)
	_, ok = out[2].(sdk.MultiRecord)
	is.True(ok) // doc-2 chunked normally despite doc-1's sibling failure
}

// --- record shapes: raw and structured payloads ---

func TestProcessor_RecordShape_RawPayload(t *testing.T) {
	is := is.New(t)
	p := newTestProcessor(t, map[string]string{"chunkSize": "1000"})
	rec := recordWithKeyAndText("doc-1", "raw payload text")

	out := p.Process(context.Background(), []opencdc.Record{rec})

	mr := out[0].(sdk.MultiRecord)
	is.Equal(len(mr), 1)
	is.Equal(chunkText(t, mr[0]), "raw payload text")
}

func TestProcessor_RecordShape_StructuredPayload(t *testing.T) {
	is := is.New(t)
	p := newTestProcessor(t, map[string]string{
		"chunkSize":   "1000",
		"inputField":  ".Payload.After.text",
		"outputField": ".Payload.After.chunk",
	})
	rec := opencdc.Record{
		Operation: opencdc.OperationCreate,
		Key:       opencdc.RawData("doc-1"),
		Payload: opencdc.Change{After: opencdc.StructuredData{
			"text":   "structured payload text",
			"author": "jane",
		}},
	}

	out := p.Process(context.Background(), []opencdc.Record{rec})

	mr := out[0].(sdk.MultiRecord)
	is.Equal(len(mr), 1)
	after, ok := mr[0].Payload.After.(opencdc.StructuredData)
	is.True(ok)
	is.Equal(after["chunk"], "structured payload text")
	is.Equal(after["author"], "jane")                  // untouched sibling field preserved
	is.Equal(after["text"], "structured payload text") // original field untouched (outputField != inputField)
}

func TestProcessor_FieldResolutionError_IsolatedPerRecord(t *testing.T) {
	is := is.New(t)
	p := newTestProcessor(t, map[string]string{
		"chunkSize":  "1000",
		"inputField": ".Payload.After.text", // record below has RawData payload, not structured
	})
	bad := recordWithKeyAndText("doc-1", "raw, not structured")
	good := opencdc.Record{
		Operation: opencdc.OperationCreate,
		Key:       opencdc.RawData("doc-2"),
		Payload:   opencdc.Change{After: opencdc.StructuredData{"text": "structured text"}},
	}

	out := p.Process(context.Background(), []opencdc.Record{bad, good})

	_, ok := out[0].(sdk.ErrorRecord)
	is.True(ok)
	_, ok = out[1].(sdk.MultiRecord)
	is.True(ok)
}

// --- config validation ---

func TestProcessor_ConfigValidate_OverlapMustBeLessThanChunkSize(t *testing.T) {
	is := is.New(t)
	p := NewProcessor()
	cfg := config.Config{"strategy": "fixed_size", "chunkSize": "10", "overlap": "10"}

	err := p.Configure(context.Background(), cfg)

	is.True(err != nil)
	var chunkErr *Error
	is.True(errors.As(err, &chunkErr))
	is.Equal(chunkErr.Code, CodeInvalidConfig)
}

func TestProcessor_ConfigValidate_UnknownStrategyRejected(t *testing.T) {
	is := is.New(t)
	p := NewProcessor()
	cfg := config.Config{"strategy": "made-up-strategy"}

	err := p.Configure(context.Background(), cfg)

	is.True(err != nil) // paramgen's inclusion validation rejects it before Config.Validate even runs
}

// --- round-trip property: chunks reconstruct the source document ---
// (design doc's Testing section: "concatenating chunks with overlap
// removed reconstructs the source text")

func TestProcessor_RoundTrip_FixedSizeWithOverlap(t *testing.T) {
	is := is.New(t)
	const overlap = 4
	p := newTestProcessor(t, map[string]string{
		"strategy": "fixed_size", "chunkSize": "17", "overlap": strconv.Itoa(overlap),
	})
	text := strings.Repeat("the quick brown fox jumps over lazy dogs ", 5)
	rec := recordWithKeyAndText("doc-1", text)

	out := p.Process(context.Background(), []opencdc.Record{rec})
	mr := out[0].(sdk.MultiRecord)
	is.True(len(mr) > 1)

	var rebuilt strings.Builder
	for i, chunkRec := range mr {
		txt := chunkText(t, chunkRec)
		if i > 0 {
			// Every chunk after the first repeats `overlap` runes from the
			// previous one - trim them back out before rejoining.
			chunkRunes := []rune(txt)
			is.True(len(chunkRunes) >= overlap)
			txt = string(chunkRunes[overlap:])
		}
		rebuilt.WriteString(txt)
	}
	is.Equal(rebuilt.String(), text)
}

func TestProcessor_RoundTrip_SentenceStrategy(t *testing.T) {
	is := is.New(t)
	p := newTestProcessor(t, map[string]string{"strategy": "sentence", "chunkSize": "1000"})
	text := "One sentence. Another sentence! A third one? And a fourth."
	rec := recordWithKeyAndText("doc-1", text)

	out := p.Process(context.Background(), []opencdc.Record{rec})
	mr := out[0].(sdk.MultiRecord)

	var rebuilt strings.Builder
	for _, chunkRec := range mr {
		rebuilt.WriteString(chunkText(t, chunkRec))
	}
	is.Equal(rebuilt.String(), text) // no overlap for this strategy - plain concatenation reconstructs exactly
}

func TestProcessor_RoundTrip_RecursiveStrategy(t *testing.T) {
	is := is.New(t)
	p := newTestProcessor(t, map[string]string{"strategy": "recursive", "chunkSize": "30"})
	text := "First paragraph with a couple sentences. Here is one more.\n\n" +
		"Second paragraph, also with content, that runs a bit longer than the first one did."
	rec := recordWithKeyAndText("doc-1", text)

	out := p.Process(context.Background(), []opencdc.Record{rec})
	mr := out[0].(sdk.MultiRecord)
	is.True(len(mr) > 1)

	var rebuilt strings.Builder
	for _, chunkRec := range mr {
		rebuilt.WriteString(chunkText(t, chunkRec))
	}
	is.Equal(rebuilt.String(), text)
}

// --- position/operation passthrough ---

func TestProcessor_ChunkRecordsPreserveSourcePositionAndOperation(t *testing.T) {
	is := is.New(t)
	p := newTestProcessor(t, map[string]string{"chunkSize": "10", "overlap": "0"})
	rec := recordWithKeyAndText("doc-1", strings.Repeat("a", 25))
	rec.Position = opencdc.Position("source-position-123")
	rec.Operation = opencdc.OperationUpdate

	out := p.Process(context.Background(), []opencdc.Record{rec})
	mr := out[0].(sdk.MultiRecord)

	for _, chunkRec := range mr {
		is.Equal(chunkRec.Position, rec.Position)
		is.Equal(chunkRec.Operation, opencdc.OperationUpdate)
	}
}

// --- sibling chunk records don't share mutable state ---

func TestProcessor_ChunkRecordsDoNotShareMetadataMap(t *testing.T) {
	is := is.New(t)
	p := newTestProcessor(t, map[string]string{"chunkSize": "10", "overlap": "0"})
	rec := recordWithKeyAndText("doc-1", strings.Repeat("a", 25))
	rec.Metadata["custom"] = "value"

	out := p.Process(context.Background(), []opencdc.Record{rec})
	mr := out[0].(sdk.MultiRecord)
	is.True(len(mr) > 1)

	mr[0].Metadata["custom"] = "mutated"
	is.Equal(mr[1].Metadata["custom"], "value") // sibling unaffected - Clone() gave each its own map
}
