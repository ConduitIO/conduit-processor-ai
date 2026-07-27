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
	"fmt"
	"os"
	"strconv"

	"github.com/conduitio/conduit-commons/config"
	"github.com/conduitio/conduit-commons/opencdc"
	sdk "github.com/conduitio/conduit-processor-sdk"
)

// Metadata keys attached to every successfully embedded record. Field
// names are stable, part of this processor's output contract (design doc
// §4).
const (
	MetadataProvider    = "ai.embedding.provider"
	MetadataModel       = "ai.embedding.model"
	MetadataDimension   = "ai.embedding.dimension"
	MetadataTokensUsed  = "ai.embedding.tokensUsed"
	MetadataTokensScope = "ai.embedding.tokensUsedScope"
)

// Values for the MetadataTokensScope metadata key — see attachEmbedding and
// [TokensScope].
const (
	metadataScopeRecord = "record"
	metadataScopeBatch  = "batch"
)

// Processor is the standalone (WebAssembly) embedding processor. It
// implements sdk.Processor's Specification/Configure/Open/Process via
// embedding sdk.UnimplementedProcessor for the methods this slice doesn't
// need to override (Teardown, MiddlewareOptions).
type Processor struct {
	sdk.UnimplementedProcessor

	config         Config
	provider       Provider
	inputResolver  sdk.ReferenceResolver
	outputResolver sdk.ReferenceResolver

	// getenv is a seam for testing CONDUIT_EMBED_PROVIDER resolution
	// without mutating process-global environment state; defaults to
	// os.Getenv in Open.
	getenv func(string) string
}

// NewProcessor constructs an embedding Processor ready for Configure.
func NewProcessor() *Processor {
	return &Processor{getenv: os.Getenv}
}

func (p *Processor) Specification() (sdk.Specification, error) {
	return sdk.Specification{
		Name:    "ai.embed",
		Summary: "Generate vector embeddings for records using a pluggable provider.",
		Description: "Reads the configured inputField, generates an embedding via the resolved provider " +
			"(openai, ollama, voyage, or cohere), and writes the " +
			"vector to outputField along with provider/model/dimension/tokensUsed metadata. " +
			"Sub-batches records within a single Process call only — see the package README's delivery-semantics " +
			"note for what happens on partial and full batch failure.",
		Version:    "v0.1.0-slice1",
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

// Validate checks cross-field invariants paramgen's per-field validations
// (required/gt=0/...) can't express, failing fast at Configure rather than
// waiting until Open. It rejects an explicitly-configured provider name that
// is not one of the four (a typo), and — as a forward-guard for the
// incremental-slice workflow (see implementedProviders) — an explicitly
// named-but-not-yet-built provider. With all four named providers now
// implemented the second guard is vacuous, but it keeps a future
// named-but-unbuilt provider failing with a clear coded error instead of a
// nil-provider panic at Open.
func (c Config) Validate() error {
	if c.Provider != "" && c.Provider != ProviderOpenAI && c.Provider != ProviderVoyage &&
		c.Provider != ProviderCohere && c.Provider != ProviderOllama {
		return &Error{
			Code:       CodeInvalidConfig,
			Message:    fmt.Sprintf("unknown provider %q", c.Provider),
			ConfigPath: ConfigProvider,
			Suggestion: "use one of: openai, voyage, cohere, ollama",
		}
	}
	if c.Provider != "" && !IsImplemented(c.Provider) {
		return errProviderNotImplemented(c.Provider)
	}
	return nil
}

// Open resolves and builds the embedding provider (unless one was already
// injected, e.g. by a test) and prepares the field resolvers. It performs
// no I/O of its own — provider resolution is pure config inspection (see
// ResolveProviderName) — so it does not use its context parameter, but
// keeps it to satisfy sdk.Processor's Open(context.Context) error contract.
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

	if p.getenv == nil {
		p.getenv = os.Getenv
	}

	// Provider may already be injected (tests). Only resolve/build it if
	// Open hasn't been given one already.
	if p.provider == nil {
		name, err := ResolveProviderName(p.config, p.getenv)
		if err != nil {
			return err
		}
		provider, err := BuildProvider(name, p.config)
		if err != nil {
			return err
		}
		p.provider = provider
	}

	return nil
}

// batchSize returns the sub-batch size: the configured ceiling clamped to
// the resolved provider's own per-request input limit (design doc §2/§4).
func (p *Processor) batchSize() int {
	size := p.config.MaxTextsPerBatch
	if size <= 0 {
		size = 1
	}
	if providerMax := p.provider.MaxBatchSize(); providerMax > 0 && providerMax < size {
		size = providerMax
	}
	return size
}

// Process sub-batches records into as few provider calls as batchSize
// allows, strictly within this one call — see doc.go and the design doc
// §4/§7 for why this never accumulates across Process invocations. The
// returned slice always has the same length as records, one
// sdk.ProcessedRecord per input record at the same index (the interface
// contract sdk.Processor.Process documents).
func (p *Processor) Process(ctx context.Context, records []opencdc.Record) []sdk.ProcessedRecord {
	out := make([]sdk.ProcessedRecord, len(records))

	size := p.batchSize()
	for start := 0; start < len(records); start += size {
		end := min(start+size, len(records))
		p.processBatch(ctx, records[start:end], out[start:end])
	}

	return out
}

// batchItem pairs a record's index (into the records/out slices Process
// was called with) with its resolved input text, for the subset of a
// sub-batch that actually reaches the provider.
type batchItem struct {
	idx  int
	text string
}

// processBatch embeds one sub-batch and writes each record's outcome into
// the matching slot of out (out is a sub-slice of Process's out, aligned
// index-for-index with records).
//
// This is the invariant-1/3 enforcement point (design doc §4):
//   - A record whose input field can't be resolved never reaches the
//     provider at all and fails on its own (ai.embedding_field_resolution_error),
//     without affecting its sub-batch siblings.
//   - If Provider.Embed itself returns an error, every record that DID
//     reach the provider in this call fails with ai.embedding_provider_error
//     — none is embedded or passed through with a placeholder vector.
//   - If Provider.Embed succeeds but returns a per-input error for some
//     inputs (Outcome.Err), that is honored 1:1: the failed inputs
//     become per-record errors, the succeeded ones proceed with their
//     real embedding. Success is never widened across the whole batch,
//     and a failed input is never silently dropped.
func (p *Processor) processBatch(ctx context.Context, records []opencdc.Record, out []sdk.ProcessedRecord) {
	items := make([]batchItem, 0, len(records))
	for i := range records {
		// Tombstones / delete intents carry no content to embed and must pass
		// through UNCHANGED so a downstream vector sink can action the delete
		// (e.g. pgvector's delete-by-source_key fan-out). Trying to embed a
		// delete would turn it into an ErrorRecord and strand the RAG delete
		// path, orphaning the vectors the delete was meant to remove.
		if records[i].Operation == opencdc.OperationDelete {
			out[i] = sdk.SingleRecord(records[i])
			continue
		}
		text, err := p.resolveInputText(&records[i])
		if err != nil {
			out[i] = sdk.ErrorRecord{Error: newFieldError(p.config.InputField, err)}
			continue
		}
		items = append(items, batchItem{idx: i, text: text})
	}

	if len(items) == 0 {
		return
	}

	texts := make([]string, len(items))
	for j, it := range items {
		texts[j] = it.text
	}

	result, err := p.provider.Embed(ctx, texts)
	if err != nil {
		// Whole-batch failure: no record in items is embedded or passed
		// through (invariant 1/3). Records that already failed field
		// resolution above keep their own, distinct error. Providers are
		// expected to return an already-coded error (see the shared
		// classifyHostedEgressError and per-provider classify*Status), but processBatch
		// wraps defensively so every record's error is guaranteed coded
		// (ai.embedding_provider_error) even if a Provider implementation
		// returns a bare error.
		batchErr := newProviderError(p.provider.Name(), err)
		for _, it := range items {
			out[it.idx] = sdk.ErrorRecord{Error: batchErr}
		}
		return
	}

	if len(result.Outcomes) != len(items) {
		// A provider that doesn't honor the 1:1 input/output contract is
		// itself a provider-shape failure. Fail the whole sub-batch
		// rather than guess a mapping — never widen a shape mismatch into
		// a partial success.
		shapeErr := newProviderError(p.provider.Name(),
			fmt.Errorf("provider returned %d results for %d inputs", len(result.Outcomes), len(items)))
		for _, it := range items {
			out[it.idx] = sdk.ErrorRecord{Error: shapeErr}
		}
		return
	}

	for j, it := range items {
		outcome := result.Outcomes[j]
		if outcome.Err != nil {
			out[it.idx] = sdk.ErrorRecord{Error: newProviderError(p.provider.Name(), outcome.Err)}
			continue
		}

		rec := records[it.idx]
		if err := p.attachEmbedding(&rec, outcome.Vector, result); err != nil {
			out[it.idx] = sdk.ErrorRecord{Error: err}
			continue
		}
		out[it.idx] = sdk.SingleRecord(rec)
	}
}

// resolveInputText resolves Config.InputField on rec and coerces it to a
// string, matching the shipped cohere.embed built-in's accepted input
// shapes (opencdc.Position, opencdc.Data, string).
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

// attachEmbedding writes vector to Config.OutputField as a native []any of
// float64 — deliberately NOT json.Marshal'd to a []byte. A []byte value
// Set on a NESTED structured field (this package's default outputField,
// ".Payload.After.vector") survives entirely in-process, but silently
// corrupts once the record crosses a protobuf boundary (the WASM guest<->
// host boundary this processor always runs behind, or a destination gRPC
// boundary downstream): opencdc.StructuredData.ToProto encodes via
// structpb (google.protobuf.Struct), which has no "bytes" leaf kind — it
// base64-encodes a []byte into a STRING, which a vector destination like
// pgvector's internal.ParseVector does not accept. A []any of float64
// elements, in contrast, becomes a structpb ListValue of NumberValues,
// which round-trips losslessly and is exactly the shape
// pgvector's internal.ParseVector documents accepting ("[]any of JSON
// numbers"). See the sibling chunk package and design doc's RAG contract
// for the composable record shape this and OutputField's default rely on.
// It also sets the provider/model/dimension/tokensUsed metadata (design
// doc §4's output contract). tokensUsed is only set when
// result.TokensScope is not TokensScopeUnknown — this package never
// estimates a token count the provider didn't report (design doc §2:
// "never estimated").
func (p *Processor) attachEmbedding(rec *opencdc.Record, vector []float32, result BatchResult) error {
	vec := make([]any, len(vector))
	for i, f := range vector {
		vec[i] = float64(f)
	}

	ref, err := p.outputResolver.Resolve(rec)
	if err != nil {
		return newFieldError(p.config.OutputField, fmt.Errorf("resolve outputField: %w", err))
	}
	if err := ref.Set(vec); err != nil {
		return newFieldError(p.config.OutputField, fmt.Errorf("set outputField: %w", err))
	}

	if rec.Metadata == nil {
		rec.Metadata = opencdc.Metadata{}
	}
	rec.Metadata[MetadataProvider] = p.provider.Name()
	rec.Metadata[MetadataModel] = result.Model
	rec.Metadata[MetadataDimension] = strconv.Itoa(len(vector))
	switch result.TokensScope {
	case TokensScopeUnknown:
		// Provider didn't report usage for this call — never estimate a
		// number it didn't give us (design doc §2: "never estimated").
	case TokensScopeRecord:
		rec.Metadata[MetadataTokensUsed] = strconv.Itoa(result.TokensUsed)
		rec.Metadata[MetadataTokensScope] = metadataScopeRecord
	case TokensScopeBatch:
		rec.Metadata[MetadataTokensUsed] = strconv.Itoa(result.TokensUsed)
		rec.Metadata[MetadataTokensScope] = metadataScopeBatch
	}

	return nil
}
