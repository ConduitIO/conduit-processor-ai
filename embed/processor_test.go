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
	"errors"
	"testing"

	"github.com/conduitio/conduit-commons/config"
	"github.com/conduitio/conduit-commons/opencdc"
	sdk "github.com/conduitio/conduit-processor-sdk"
	"github.com/matryer/is"
)

// fakeProvider is the Provider-seam test double used throughout this file
// — it lets processor batching/partial-failure logic be exercised without
// a real egress call or WASM host, per the task's "mock the egress call at
// the Provider/transport seam" instruction. The openai.go adapter has its
// own tests (openai_test.go) that mock one level lower, at egress.Do
// itself.
type fakeProvider struct {
	name     string
	maxBatch int
	calls    [][]string
	resultFn func(inputs []string) (BatchResult, error)
}

func (f *fakeProvider) Name() string      { return f.name }
func (f *fakeProvider) MaxBatchSize() int { return f.maxBatch }
func (f *fakeProvider) Embed(_ context.Context, inputs []string) (BatchResult, error) {
	f.calls = append(f.calls, append([]string(nil), inputs...))
	return f.resultFn(inputs)
}

// allSucceed returns a resultFn that succeeds every input with a
// fixed-dimension vector and batch-scoped token usage.
func allSucceed(dim, tokens int) func([]string) (BatchResult, error) {
	return func(inputs []string) (BatchResult, error) {
		outcomes := make([]Outcome, len(inputs))
		for i := range inputs {
			vec := make([]float32, dim)
			outcomes[i] = Outcome{Vector: vec}
		}
		scope := TokensScopeBatch
		if len(inputs) == 1 {
			scope = TokensScopeRecord
		}
		return BatchResult{
			Outcomes: outcomes, Model: "fake-model", Dimension: dim,
			TokensUsed: tokens, TokensScope: scope,
		}, nil
	}
}

func newTestProcessor(t *testing.T, cfgOverrides map[string]string, provider Provider) *Processor {
	t.Helper()
	is := is.New(t)

	p := NewProcessor()
	cfg := config.Config{}
	for k, v := range cfgOverrides {
		cfg[k] = v
	}
	is.NoErr(p.Configure(context.Background(), cfg))
	p.provider = provider // preset so Open doesn't need real resolution/egress
	is.NoErr(p.Open(context.Background()))
	return p
}

// recordWithText builds a record matching the composable RAG record shape
// (design doc / RAG contract) a chunk record actually carries: text under a
// named "text" field of a StructuredData payload, never raw bytes — this is
// what Config.InputField's new default (".Payload.After.text") expects.
func recordWithText(text string) opencdc.Record {
	return opencdc.Record{
		Metadata: opencdc.Metadata{},
		Payload:  opencdc.Change{After: opencdc.StructuredData{"text": text}},
	}
}

// --- Sub-batching math ---

func TestProcessor_BatchSizeClampedToProviderMax(t *testing.T) {
	is := is.New(t)
	fp := &fakeProvider{name: "fake", maxBatch: 2, resultFn: allSucceed(3, 0)}
	p := newTestProcessor(t, map[string]string{"maxTextsPerBatch": "5"}, fp)

	is.Equal(p.batchSize(), 2) // provider's ceiling (2) wins over config (5)

	records := make([]opencdc.Record, 5)
	for i := range records {
		records[i] = recordWithText("text")
	}
	out := p.Process(context.Background(), records)

	is.Equal(len(out), 5)
	is.Equal(len(fp.calls), 3) // ceil(5/2) sub-batch calls
	is.Equal(len(fp.calls[0]), 2)
	is.Equal(len(fp.calls[1]), 2)
	is.Equal(len(fp.calls[2]), 1)
	for _, pr := range out {
		_, ok := pr.(sdk.SingleRecord)
		is.True(ok)
	}
}

func TestProcessor_ConfigBatchSizeUsedWhenSmallerThanProviderMax(t *testing.T) {
	is := is.New(t)
	fp := &fakeProvider{name: "fake", maxBatch: 100, resultFn: allSucceed(3, 0)}
	p := newTestProcessor(t, map[string]string{"maxTextsPerBatch": "2"}, fp)

	is.Equal(p.batchSize(), 2)

	records := make([]opencdc.Record, 3)
	for i := range records {
		records[i] = recordWithText("text")
	}
	p.Process(context.Background(), records)
	is.Equal(len(fp.calls), 2)
	is.Equal(len(fp.calls[0]), 2)
	is.Equal(len(fp.calls[1]), 1)
}

func TestProcessor_NeverAccumulatesAcrossProcessCalls(t *testing.T) {
	is := is.New(t)
	fp := &fakeProvider{name: "fake", maxBatch: 96, resultFn: allSucceed(3, 0)}
	p := newTestProcessor(t, map[string]string{"maxTextsPerBatch": "96"}, fp)

	// Two separate Process calls, one record each. If batching ever
	// accumulated across calls (the design explicitly forbids this — see
	// design doc §4/§7), the second call might wait for the first's
	// leftover instead of embedding immediately.
	out1 := p.Process(context.Background(), []opencdc.Record{recordWithText("first")})
	out2 := p.Process(context.Background(), []opencdc.Record{recordWithText("second")})

	is.Equal(len(out1), 1)
	is.Equal(len(out2), 1)
	is.Equal(len(fp.calls), 2) // one provider call per Process call, never merged
	is.Equal(fp.calls[0], []string{"first"})
	is.Equal(fp.calls[1], []string{"second"})
}

// --- Invariant 1/3: partial-batch and full-batch failure handling ---

// TestProcessor_TombstonePassesThroughUnembedded is the RAG-delete-path
// regression: a delete tombstone (the chunking processor's re-emitted delete
// intent) must pass through the embedding stage UNCHANGED — never embedded,
// never turned into an ErrorRecord — so it reaches the vector sink's
// delete-by-source_key fan-out. Embedding a delete would orphan the vectors the
// delete was meant to remove.
func TestProcessor_TombstonePassesThroughUnembedded(t *testing.T) {
	is := is.New(t)

	var gotInputs [][]string
	fp := &fakeProvider{
		name:     "fake",
		maxBatch: 96,
		resultFn: func(inputs []string) (BatchResult, error) {
			gotInputs = append(gotInputs, append([]string(nil), inputs...))
			return allSucceed(len(inputs), 0)(inputs)
		},
	}
	p := newTestProcessor(t, nil, fp)

	tombstone := opencdc.Record{
		Operation: opencdc.OperationDelete,
		Key:       opencdc.RawData("orders:42"),
		Metadata:  opencdc.Metadata{"ai.chunk.source_key": "orders:42"},
	}
	out := p.Process(context.Background(), []opencdc.Record{
		recordWithText("embed me"),
		tombstone,
	})

	is.Equal(len(out), 2)

	// The content record is embedded.
	_, ok := out[0].(sdk.SingleRecord)
	is.True(ok)

	// The tombstone passes through unchanged: SingleRecord, still a delete, no
	// embedding written, and NOT an ErrorRecord.
	passed, ok := out[1].(sdk.SingleRecord)
	is.True(ok)
	is.Equal(opencdc.Record(passed).Operation, opencdc.OperationDelete)
	is.Equal(opencdc.Record(passed).Metadata["ai.chunk.source_key"], "orders:42")

	// The provider was only ever handed the one content record — the tombstone
	// never reached it.
	is.Equal(len(gotInputs), 1)
	is.Equal(gotInputs[0], []string{"embed me"})
}

func TestProcessor_FullBatchFailure_NoPassthrough(t *testing.T) {
	is := is.New(t)
	wantErr := errors.New("provider unreachable")
	fp := &fakeProvider{
		name:     "fake",
		maxBatch: 96,
		resultFn: func([]string) (BatchResult, error) { return BatchResult{}, wantErr },
	}
	p := newTestProcessor(t, nil, fp)

	records := []opencdc.Record{recordWithText("a"), recordWithText("b"), recordWithText("c")}
	out := p.Process(context.Background(), records)

	is.Equal(len(out), 3)
	for _, pr := range out {
		errRec, ok := pr.(sdk.ErrorRecord)
		is.True(ok) // every record in a failed batch must be an ErrorRecord, never SingleRecord
		var perr *Error
		is.True(errors.As(errRec.Error, &perr))
		is.Equal(perr.Code, CodeProviderError)
		is.True(errors.Is(errRec.Error, wantErr))
	}
}

func TestProcessor_PartialBatchFailure_HonoredOneToOne(t *testing.T) {
	is := is.New(t)
	recordErr := errors.New("malformed input at index 1")
	fp := &fakeProvider{
		name:     "fake",
		maxBatch: 96,
		resultFn: func(inputs []string) (BatchResult, error) {
			outcomes := make([]Outcome, len(inputs))
			for i := range inputs {
				if i == 1 {
					outcomes[i] = Outcome{Err: recordErr}
					continue
				}
				outcomes[i] = Outcome{Vector: []float32{0.1, 0.2}}
			}
			return BatchResult{Outcomes: outcomes, Model: "fake-model", Dimension: 2, TokensScope: TokensScopeBatch, TokensUsed: 10}, nil
		},
	}
	p := newTestProcessor(t, nil, fp)

	records := []opencdc.Record{recordWithText("ok-a"), recordWithText("bad"), recordWithText("ok-b")}
	out := p.Process(context.Background(), records)

	is.Equal(len(out), 3)

	_, ok0 := out[0].(sdk.SingleRecord)
	is.True(ok0) // succeeded record proceeds

	errRec1, ok1 := out[1].(sdk.ErrorRecord)
	is.True(ok1) // failed record is reported individually
	var perr *Error
	is.True(errors.As(errRec1.Error, &perr))
	is.Equal(perr.Code, CodeProviderError)
	is.True(errors.Is(errRec1.Error, recordErr))

	_, ok2 := out[2].(sdk.SingleRecord)
	is.True(ok2) // the failure at index 1 does not widen to index 2

	// Success is not silently widened: index 1 never got a placeholder
	// vector, and index 0/2's success is not affected by index 1's error.
}

func TestProcessor_ProviderShapeMismatch_FailsWholeSubBatch(t *testing.T) {
	is := is.New(t)
	fp := &fakeProvider{
		name:     "fake",
		maxBatch: 96,
		resultFn: func([]string) (BatchResult, error) {
			// Returns fewer outcomes than inputs — a provider-shape bug.
			return BatchResult{Outcomes: []Outcome{{Vector: []float32{0.1}}}}, nil
		},
	}
	p := newTestProcessor(t, nil, fp)

	out := p.Process(context.Background(), []opencdc.Record{recordWithText("a"), recordWithText("b")})
	for _, pr := range out {
		_, ok := pr.(sdk.ErrorRecord)
		is.True(ok) // never guess a mapping — fail closed, not widen to success
	}
}

func TestProcessor_FieldResolutionError_IsolatedPerRecord(t *testing.T) {
	is := is.New(t)
	fp := &fakeProvider{name: "fake", maxBatch: 96, resultFn: allSucceed(2, 0)}
	p := newTestProcessor(t, nil, fp)

	badRecord := opencdc.Record{Metadata: opencdc.Metadata{}} // Payload.After is nil opencdc.Data
	records := []opencdc.Record{recordWithText("good-a"), badRecord, recordWithText("good-b")}
	out := p.Process(context.Background(), records)

	is.Equal(len(out), 3)
	_, ok0 := out[0].(sdk.SingleRecord)
	is.True(ok0)

	errRec, ok1 := out[1].(sdk.ErrorRecord)
	is.True(ok1)
	var perr *Error
	is.True(errors.As(errRec.Error, &perr))
	is.Equal(perr.Code, CodeFieldResolution)

	_, ok2 := out[2].(sdk.SingleRecord)
	is.True(ok2)

	// The bad record must never reach the provider at all.
	is.Equal(len(fp.calls), 1)
	is.Equal(fp.calls[0], []string{"good-a", "good-b"})
}

// --- Output contract: metadata + native vector shape round-trip ---

// TestProcessor_AttachEmbedding_MetadataAndVectorRoundTrip proves the
// composable RAG record shape (design doc / RAG contract) with this
// package's defaults left untouched: InputField reads ".Payload.After.text"
// (recordWithText's shape) and OutputField writes ".Payload.After.vector" —
// leaving "text" in place alongside the new "vector" field, and writing the
// vector as a native []any of float64 (never json.Marshal'd bytes — see
// attachEmbedding's doc comment for why a []byte on a nested field silently
// corrupts across a protobuf boundary).
func TestProcessor_AttachEmbedding_MetadataAndVectorRoundTrip(t *testing.T) {
	is := is.New(t)
	fp := &fakeProvider{name: "openai", maxBatch: 96, resultFn: allSucceed(4, 123)}
	p := newTestProcessor(t, nil, fp)

	out := p.Process(context.Background(), []opencdc.Record{recordWithText("hello")})
	is.Equal(len(out), 1)
	single, ok := out[0].(sdk.SingleRecord)
	is.True(ok)
	rec := opencdc.Record(single)

	is.Equal(rec.Metadata[MetadataProvider], "openai")
	is.Equal(rec.Metadata[MetadataModel], "fake-model")
	is.Equal(rec.Metadata[MetadataDimension], "4")
	is.Equal(rec.Metadata[MetadataTokensUsed], "123")
	is.Equal(rec.Metadata[MetadataTokensScope], "record") // single-input sub-batch

	after, ok := rec.Payload.After.(opencdc.StructuredData)
	is.True(ok)
	is.Equal(after["text"], "hello") // original text preserved alongside the vector

	vecAny, ok := after["vector"].([]any)
	is.True(ok) // native []any, never a JSON-encoded []byte
	is.Equal(len(vecAny), 4)
	for _, e := range vecAny {
		_, ok := e.(float64)
		is.True(ok) // every element is a float64, matching pgvector's internal.ParseVector's accepted shape
	}
}

// --- Configure/Open wiring ---

func TestProcessor_Open_ResolvesRealOpenAIProviderFromConfig(t *testing.T) {
	is := is.New(t)
	p := NewProcessor()
	is.NoErr(p.Configure(context.Background(), config.Config{
		"openai.authSecretRef": "my-secret",
		"model":                "text-embedding-3-small",
	}))
	is.NoErr(p.Open(context.Background()))
	is.Equal(p.provider.Name(), ProviderOpenAI)
}

func TestProcessor_Configure_RejectsUnimplementedProviderEarly(t *testing.T) {
	is := is.New(t)
	p := NewProcessor()
	err := p.Configure(context.Background(), config.Config{"provider": "cohere"})
	is.True(err != nil)
	var perr *Error
	is.True(errors.As(err, &perr))
	is.Equal(perr.Code, CodeProviderNotImplemented)
}

func TestProcessor_Configure_AppliesDefaults(t *testing.T) {
	is := is.New(t)
	p := NewProcessor()
	is.NoErr(p.Configure(context.Background(), config.Config{"openai.authSecretRef": "k"}))
	is.Equal(p.config.MaxTextsPerBatch, 96)
	is.Equal(p.config.InputField, ".Payload.After.text")
	is.Equal(p.config.OutputField, ".Payload.After.vector")
	is.Equal(p.config.MaxRetries, 5)
}

func TestProcessor_Specification(t *testing.T) {
	is := is.New(t)
	p := NewProcessor()
	spec, err := p.Specification()
	is.NoErr(err)
	is.Equal(spec.Name, "ai.embed")
	is.True(len(spec.Parameters) > 0)
}
