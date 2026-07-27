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

// Package embed implements Conduit's standalone (WebAssembly) embedding
// processor: it turns a batch of records into the same records carrying a
// vector-embedding payload field, calling a pluggable provider (OpenAI,
// Voyage, Cohere, or a local Ollama server) through the host-mediated
// network-egress capability described in
// docs/design-documents/20260726-wasm-host-egress-capability.md and used by
// this processor per
// docs/design-documents/20260724-ai-pipeline-components.md §4 (both in
// ConduitIO/conduit).
//
// # Slice 1 scope
//
// This is the first reviewable slice of the embedding processor.
// [ProviderOpenAI] and [ProviderOllama] adapters are implemented end to
// end; [ProviderVoyage] and [ProviderCohere] are named constants with a
// wired resolution seam (config fields, auto-detection, ambiguity
// checking) but selecting one yields a coded
// "ai.embedding_provider_not_implemented" error, not a working call. See
// the package README's "Slicing note" for the plan to fill in the
// remaining two providers, acceptance tests, and the bundle end-to-end
// test in later slices.
//
// # Why the guest hand-rolls JSON instead of reusing a vendor SDK
//
// The design doc's provider table (§2) frames each provider's zero-new-
// dependency justification around reusing an already-vendored Go client
// (go-openai for OpenAI, cohere-go for Cohere) the way the core engine's
// built-in cohere.embed and openai.embeddings processors already do. That
// justification does not carry over to this repo: this processor is a
// standalone WASM guest (WASI Preview 1, no socket API — the entire reason
// the host-egress capability in §1 of the design doc exists), and every
// vendor SDK does its own net/http dialing internally. A vendor SDK cannot
// run inside the guest sandbox at all, regardless of dependency-budget
// arguments. The only coherent implementation is: this package hand-rolls
// each provider's request/response JSON as thin structs (the endpoints are
// simple JSON-in/JSON-out) and calls [egress.Do] for the actual transport,
// which the host performs under its allowlist/DNS-rebinding/timeout/
// size-cap policy. See [newOpenAIProvider] and openai.go for the shape;
// [newOllamaProvider] and ollama.go follow the same shape for Ollama's
// single-input-per-call /api/embeddings endpoint.
//
// # Batching is strictly within one Process call
//
// [Processor.Process] sub-batches only the records the engine hands it in
// the call currently executing into as few [Provider.Embed] calls as the
// provider's batch-size ceiling allows (default 96, config
// maxTextsPerBatch, mirroring the core engine's built-in cohere.embed
// processor). It never accumulates records across separate Process
// invocations — see the design doc §4 and §7 for why cross-call
// accumulation is rejected outright as an ack-correctness hazard, not
// merely deferred.
//
// # Partial-batch failure is never widened, never silently dropped
//
// If a [Provider.Embed] call fails outright, every record that reached that
// call is failed (an [sdk.ErrorRecord]) — none is embedded or passed
// through with a placeholder vector. If a provider's response reports some
// inputs succeeded and others failed within one call (a real API shape,
// e.g. one malformed input in an otherwise-good batch), this package
// honors that 1:1: each [Outcome] maps back to exactly the record at
// its input index, succeeded records proceed with their real embedding,
// failed ones become per-record [sdk.ErrorRecord]s so the engine's DLQ/
// retry policy applies. See [Processor.processBatch] in processor.go —
// this is the invariant-1/3 correctness point the design doc calls out as
// load-bearing, and it is covered by TestProcessor_PartialBatchFailure and
// TestProcessor_FullBatchFailure in processor_test.go.
package embed
