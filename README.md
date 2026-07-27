# conduit-processor-ai

Conduit processors for AI pipelines: chunking + embedding with pluggable providers (OpenAI,
Voyage, Ollama, Cohere). Part of Conduit v0.20 (WS8) —
see `docs/design-documents/20260724-ai-pipeline-components.md` in `ConduitIO/conduit` for the
full design.

Two processors, two packages, one repo (design doc §3/§4, Open Questions §1 — kept together
since they're always deployed as a pair in the canonical RAG pipeline):

- **`ai.chunk`** (`chunk/`) — splits a record's text into N chunk records. Pure in-memory, no
  host capability. See "ai.chunk — chunking processor" below.
- **`ai.embed`** (`embed/`) — generates vector embeddings via a pluggable provider, using the
  host-mediated network-egress capability. See "ai.embed — embedding processor" below.

## `ai.chunk` — chunking processor (Slice 1)

The RAG pipeline's first stage: `Postgres CDC → chunk → embed → pgvector`. This is the
**first reviewable slice**: all three strategies (`fixed_size`, `sentence`, `recursive`), the
fan-out/chunk_id/metadata contract, and tombstone handling are implemented and tested. Not in
this slice: acceptance tests and the end-to-end RAG-sync template (see "Slicing note" below).

### Strategies

Selected via the `strategy` config key; see `chunk/doc.go` and `chunk/span.go` for the full
algorithm documentation.

| Strategy | What it does | Honors `overlap`? |
| --- | --- | --- |
| `fixed_size` (default) | A sliding window of `chunkSize` runes, advancing by `chunkSize - overlap` runes each step. | Yes — the only strategy that does. |
| `sentence` | Splits on sentence-ending punctuation (`.`, `!`, `?`) followed by whitespace or end-of-text — a dependency-free heuristic, not an NLP tokenizer (no abbreviation handling). Sentences are packed greedily up to `chunkSize` without ever splitting one; a single sentence longer than `chunkSize` becomes its own oversized chunk rather than being cut mid-sentence. | No — natural boundaries, never repeated. |
| `recursive` | Tries paragraph (`\n\n`) boundaries first; any piece still over `chunkSize` is recursively split on sentence boundaries, then word (whitespace) boundaries, then — only if a single word still exceeds `chunkSize` — hard-split at the rune level. Unlike `sentence` alone, **every** chunk is guaranteed to fit `chunkSize`. | No — natural boundaries, never repeated. |

**Offsets and `chunkSize` are Unicode characters (runes), never bytes.** Every strategy converts
its input to `[]rune` exactly once and only ever slices that rune slice, so a chunk boundary can
never land inside a multi-byte character — see `chunk/span_test.go`'s unicode tests.

### Fan-out and the metadata contract (design doc §3/§6)

One input record produces **zero, one, or many** output records:

- **Empty input text → zero chunks** (`sdk.MultiRecord{}`, equivalent to a filter — nothing to
  embed downstream).
- **Non-empty text → one output record per chunk**, in order. Each chunk record carries:
  - The chunk's text under a NAMED `"text"` field of a `StructuredData` payload (`outputField`,
    default `.Payload.After.text`) — never raw bytes. This is the composable RAG record shape: the
    sibling `ai.embed` processor's own default `inputField` (`.Payload.After.text`) reads this
    field directly, and its default `outputField` (`.Payload.After.vector`) adds the embedding
    vector alongside it without clobbering the text, so the canonical
    `chunk → embed → pgvector` pipeline needs zero `inputField`/`outputField` configuration by
    default (see `ai.embed`'s config reference and the example pipeline below).
  - `Key` set to the chunk's `chunk_id` (see below) — gives a destination connector's default
    upsert-by-`Key` behavior the right identity for free.
  - Metadata (**exact keys — this is the cross-component contract the pgvector destination
    consumes**):

    | Key | Value |
    | --- | --- |
    | `ai.chunk.id` | The deterministic `chunk_id`: `{source_record_key}:{chunk_index}` (0-based). |
    | `ai.chunk.source_key` | The source record's `Key`, stringified. |
    | `ai.chunk.index` | The chunk's 0-based index among its source record's chunks. |
    | `ai.chunk.offset` | The chunk's start offset in the source document, in runes. |
    | `ai.chunk.length` | The chunk's length, in runes. |

**`chunk_id` is deterministic, never random or time-based** — derived solely from the source
record's `Key` and the chunk's 0-based index. Two independent runs over an identical source
record produce byte-for-byte identical `chunk_id`s; this is what makes the downstream pgvector
upsert idempotent under retry/redelivery (design doc §6). `TestProcessor_DeterministicChunkIDs`
in `chunk/processor_test.go` is the test that proves this property — it runs two fresh
`Processor` instances (simulating two independent attempts) over the same record and asserts
identical `chunk_id`s in the same order.

**A source record with no `Key` cannot produce a stable `chunk_id`.** Rather than fabricate one
(which would silently break the idempotency guarantee above), this is a coded, per-record error
(`ai.chunk_missing_source_key`) routed to the pipeline's DLQ/error policy — never a silent drop,
never a random fallback ID.

### Tombstones / delete-intents

A delete of the source record (`Operation == OperationDelete`) is **never chunked.** Instead the
processor re-emits the exact same record — same `Operation`, `Key`, `Payload` (still a normal
opencdc delete any destination understands on its own) — with one addition:
`ai.chunk.source_key` metadata naming the deleted row.

This is deliberately **not** a per-`chunk_id` delete: a single source row can have produced a
different number of chunks at different points in its history (the document shrank or grew
between edits), so this processor doesn't know — and doesn't guess — the full set of `chunk_id`s
that need deleting. The vector destination resolves "every `chunk_id` ever derived from this
`source_key`" itself by matching on the `ai.chunk.source_key` column (design doc §5). A tombstone
with no `Key` is a coded error (`ai.chunk_missing_source_key`), the same as any other record —
never silently dropped.

### Configuration reference

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `strategy` | string | `fixed_size` | `fixed_size` \| `sentence` \| `recursive`. |
| `chunkSize` | int | `1000` | Target maximum chunk size, in runes. |
| `overlap` | int | `100` | Trailing runes repeated at the start of the next chunk. Only honored by `fixed_size`; must be `< chunkSize` for that strategy (`Config.Validate`, enforced at `Configure` time). |
| `inputField` | string | `.Payload.After` | Record field read as the text to chunk. |
| `outputField` | string | `.Payload.After.text` | Record field each chunk's text is written to, under a named `"text"` key of a `StructuredData` payload — see the fan-out section above. |

### Example pipeline

```yaml
version: "2.2"
pipelines:
  - id: rag-chunk-example
    status: running
    connectors:
      - id: source
        type: source
        plugin: builtin:generator
        settings:
          format.type: raw
          format.options.text: "a longer document to split into chunks..."
      - id: destination
        type: destination
        plugin: builtin:log
    processors:
      - id: chunk
        plugin: standalone:ai-chunk # installed via `conduit processors install ai-chunk@<version>`
        settings:
          strategy: recursive
          chunkSize: "500"
      - id: embed
        plugin: standalone:ai-embed
        settings:
          openai.authSecretRef: openai-api-key
          model: text-embedding-3-small
          # inputField/outputField are intentionally left at their defaults here:
          # ai.chunk's default outputField (.Payload.After.text) and ai.embed's
          # default inputField (.Payload.After.text) / outputField
          # (.Payload.After.vector) already compose — no field configuration
          # needed to wire chunk -> embed -> a pgvector destination (whose own
          # default vectorField is "vector").
```

### Slicing note — what's deliberately not in this slice

- **Acceptance tests.** This slice has unit tests covering every strategy (including overlap,
  boundary cases, and unicode/multibyte correctness), determinism, fan-out, tombstone handling,
  and record shapes — a `conduit-connector-sdk`-style acceptance suite is a follow-up.
- **The bundle end-to-end test.** Postgres CDC → chunk → embed → pgvector, CI-tested with records
  asserted at the vector store (design doc §8/Testing) — depends on `conduit-connector-pgvector`,
  out of scope here.
- **Token-count-aware chunking.** `chunkSize` is a rune count, not a model-specific token count —
  matching the design doc's "character count" framing for `fixed_size` (§3). A token-aware
  variant is not in this slice.

## `ai.embed` — embedding processor (Slice 1: OpenAI + Ollama)

This is the **first reviewable slice** of the embedding processor. Scope:

- The `ai.embed` standalone-WASM processor, using `conduit-processor-sdk`'s host-mediated
  network-egress capability (`egress.Do`) for outbound HTTP — a WASI Preview 1 guest has no
  socket API of its own.
- Two working providers: **OpenAI** (`POST /v1/embeddings`) and **Ollama** (local,
  `POST /api/embeddings`, one input per call — see the config table's `ollama.baseURL` row).
- The `Provider` seam (config, resolution, ambiguity detection) is wired for **all four**
  providers the design doc names (OpenAI, Voyage, Cohere, Ollama), but Voyage and Cohere don't
  have a working implementation yet — selecting either yields a coded
  `ai.embedding_provider_not_implemented` error. See "Slicing note" below.
- The **chunking processor** is not part of this slice.

### Why this hand-rolls JSON instead of reusing a vendor SDK

The design doc's provider table frames OpenAI/Cohere's zero-new-dependency justification around
reusing the already-vendored `go-openai`/`cohere-go` clients the core engine's built-in
`openai.embeddings`/`cohere.embed` processors use. That justification does not carry over here:
this processor is a **standalone WASM guest** (WASI Preview 1, no socket API — the entire reason
the host-egress capability exists), and every vendor SDK does its own `net/http` dialing
internally, which cannot run inside the guest sandbox. The only coherent implementation is:
hand-roll each provider's request/response JSON as thin structs (the endpoints are simple
JSON-in/JSON-out) and call `egress.Do` for the actual transport, which the host performs under
its allowlist/DNS-rebinding/timeout/size-cap policy. See `embed/openai.go`, `embed/ollama.go`, and
`embed/doc.go`.

### Delivery semantics — read before wiring this into a pipeline

- **Batching is strictly within one `Process` call, never across calls.** The processor
  sub-batches only the records the engine hands it in the call currently executing into as few
  provider calls as `maxTextsPerBatch` (clamped to the provider's own limit) allows. It never
  accumulates records across separate `Process` invocations — see `embed/doc.go` for why
  cross-call accumulation is an ack-correctness hazard, not merely a simplification.
- **A full-batch provider failure fails every record in that batch.** No record is embedded or
  passed through with a placeholder vector; every record becomes an `ErrorRecord` carrying a
  coded `ai.embedding_provider_error`, so the pipeline's configured DLQ/retry policy applies.
  Nothing is silently dropped or acked.
- **A partial-batch result is honored 1:1.** If a provider's response indicates some inputs
  succeeded and others failed within the same call, this processor never widens the failure to
  the whole batch and never silently drops the failed ones — each record's outcome is reported
  individually.
- **A record whose configured `inputField` can't be resolved never reaches the provider at all**
  and fails on its own (`ai.embedding_field_resolution_error`), without affecting its sub-batch
  siblings.
- **429 / rate-limit responses get bounded exponential backoff**, honoring a `Retry-After` header
  when the provider sends one. Exhausting `maxRetries` surfaces `ai.embedding_provider_error` —
  the batch is not silently dropped or acked. A non-429 4xx (e.g. a bad API key) is **not**
  retried — it fails fast so a misconfigured pipeline doesn't burn its retry budget.
- **`tokensUsed` metadata is never estimated.** OpenAI's `/v1/embeddings` reports token usage
  once per call, not per input, so for a sub-batch of more than one record the `tokensUsed`
  metadata is the **whole sub-batch's** token count, verbatim from the provider, tagged
  `ai.embedding.tokensUsedScope: batch` — not divided per record. Summing `tokensUsed` across
  records in the same sub-batch will over-count; group by `(provider, model, tokensUsedScope)`
  and dedupe by batch if you need an accurate per-pipeline total. A single-record sub-batch gets
  `tokensUsedScope: record`, which is exact for that one record. Ollama's `/api/embeddings`
  reports no usage figure at all, so an ollama-embedded record gets **no** `tokensUsed`/
  `tokensUsedScope` metadata — never a fabricated `0`.
- **Ollama accepts exactly one input per call.** `maxTextsPerBatch` is clamped to Ollama's
  provider-reported ceiling of 1 (`ollamaProvider.MaxBatchSize`), so the sub-batcher issues one
  `POST /api/embeddings` call per record for this provider — expected behavior given the vendor
  API's shape, not a missed batching optimization.

### Configuration reference

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `provider` | string | _(empty)_ | Explicit provider: `openai` \| `voyage` \| `cohere` \| `ollama`. See Resolution below. |
| `model` | string | _(empty, required once resolved)_ | Provider embedding model, e.g. `text-embedding-3-small`. |
| `inputField` | string | `.Payload.After.text` | Record field read as the text to embed — matches `ai.chunk`'s default `outputField` so the two processors compose with no field configuration. |
| `outputField` | string | `.Payload.After.vector` | Record field the embedding vector (a native numeric array, never a JSON-encoded byte string) is written to, alongside the preserved `inputField` text. Also the field `conduit-connector-pgvector`'s destination reads by default (its own `vectorField` config defaults to `"vector"`). |
| `maxTextsPerBatch` | int | `96` | Sub-batch ceiling within one `Process` call, clamped to the provider's own per-request limit. |
| `requestTimeout` | duration | `30s` | Per host-mediated HTTP call deadline. |
| `maxRetries` | int | `5` | Max retry attempts per sub-batch on 429/5xx before failing with `ai.embedding_provider_error`. |
| `retryBackoff.min` | duration | `500ms` | Backoff floor when no `Retry-After` header is present. |
| `retryBackoff.max` | duration | `30s` | Backoff ceiling. |
| `retryBackoff.factor` | float | `2` | Exponential backoff multiplier. |
| `openai.authSecretRef` | string | _(empty)_ | Name of a host-managed secret holding the OpenAI API key. **Never a raw key value** — see Credentials below. |
| `openai.baseURL` | string | `https://api.openai.com` | OpenAI API base URL override. Must be within the pipeline's egress allowlist. |
| `voyage.authSecretRef` | string | _(empty)_ | Wired for resolution/ambiguity detection; `voyage` provider not yet implemented. |
| `cohere.authSecretRef` | string | _(empty)_ | Wired for resolution/ambiguity detection; `cohere` provider not yet implemented. |
| `ollama.baseURL` | string | _(empty, defaults to `http://localhost:11434` once ollama is selected)_ | Local Ollama server base URL. No auth secret — Ollama takes no API key. Must resolve within the pipeline's egress allowlist as an explicit `(IP,port)` carve-out for a loopback/private target. |

### Provider resolution

Mirrors `conduit generate`'s provider resolution (design doc §2):

1. **Explicit config**: `provider` key.
2. **Explicit via environment**: `CONDUIT_EMBED_PROVIDER`.
3. **Auto-detect exactly one candidate**: a provider is a candidate if its
   `<provider>.authSecretRef` (or, for Ollama, `ollama.baseURL`) config field is set. Zero
   candidates → `ai.no_provider_configured`. More than one → `ai.ambiguous_provider_configuration`.

Auto-detection is judged on **config-level signals**, not credential values or environment
variables for a provider's own API key — see the note on credentials below for why.

### Credentials

`openai.authSecretRef` names a secret; Conduit's host resolves it and injects it as the
`Authorization` header immediately before dispatch. **The processor never sees the raw API key** —
this is a property of the underlying host-egress capability (`conduit-processor-sdk`'s `egress`
package), not something this processor opts into. There is no config path for a guest-supplied
credential value.

`ollama` has no credential config at all: a local Ollama server takes no API key, so
`ollamaProvider` never sets `AuthSecretRef` on its `egress.Request`. Reachability instead depends
on the pipeline's egress allowlist granting the server's `(IP,port)` as an explicit carve-out.

### Output

On success, each record gets:

- The embedding vector as a native array of `float64` elements (`[]any` in Go terms — **not** a
  `json.Marshal`'d byte string) at `outputField`, default `.Payload.After.vector`, alongside the
  original text still at `inputField` (default `.Payload.After.text`). The native-array shape is
  deliberate: a `[]byte` value set on a *nested* structured field survives entirely in-process but
  is silently corrupted once the record crosses a protobuf boundary (the WASM guest↔host boundary
  this processor always runs behind, or a destination gRPC boundary downstream) — `structpb`
  (`google.protobuf.Struct`, which `opencdc.StructuredData.ToProto` uses) has no "bytes" leaf kind
  and base64-encodes a `[]byte` into a `STRING` instead, which a vector destination like
  `conduit-connector-pgvector`'s `internal.ParseVector` does not accept. A `[]any` of `float64`
  becomes a `structpb` `ListValue` of `NumberValue`s, which round-trips losslessly and is exactly
  the shape `ParseVector` documents accepting.
- Metadata: `ai.embedding.provider`, `ai.embedding.model`, `ai.embedding.dimension`,
  `ai.embedding.tokensUsed` (only when the provider reported usage — see the tokensUsed note
  above), `ai.embedding.tokensUsedScope` (`record` or `batch`).

### Example pipeline

```yaml
version: "2.2"
pipelines:
  - id: rag-embed-example
    status: running
    connectors:
      - id: source
        type: source
        plugin: builtin:generator
        settings:
          format.type: structured
          format.options.text: "sample chunk text"
      - id: destination
        type: destination
        plugin: builtin:log
    processors:
      - id: embed
        plugin: standalone:ai-embed # installed via `conduit processors install ai-embed@<version>`
        settings:
          openai.authSecretRef: openai-api-key # configured separately as a Conduit secret
          model: text-embedding-3-small
          # inputField/outputField left at their defaults (.Payload.After.text /
          # .Payload.After.vector) — set explicitly only if the upstream
          # processor doesn't follow ai.chunk's default output shape.
          maxTextsPerBatch: "96"
```

## Building

```sh
# Host-arch build, for running tests (both processors):
go build ./...

# The real target — standalone WASM, one binary per processor:
GOOS=wasip1 GOARCH=wasm go build -tags wasm -o chunking.wasm ./cmd/chunking
GOOS=wasip1 GOARCH=wasm go build -tags wasm -o embedding.wasm ./cmd/embedding

go vet ./...
go test -race ./...
```

`ai.chunk` (`chunk/`) needs no host capability and no `egress` dependency — only `ai.embed`
(`embed/`) uses the network-egress capability described below.

## Development note: the `go.mod` replace directive

`go.mod` currently has a `replace` pointing at a local, unreleased checkout of
`conduit-processor-sdk`'s `feat/wasm-host-egress` branch — the `egress` package `ai.embed`
depends on isn't tagged yet (`ai.chunk` doesn't use it and is unaffected by this note).
**This must be repointed to a tagged `conduit-processor-sdk` release before this repo's PR
merges.** A `.golangci.yml` exclusion scoped to `go.mod` documents this; remove both the
`replace` and the exclusion together once a tagged SDK release ships the `egress` package.

### Slicing note — what's deliberately not in this `ai.embed` slice

- **`voyage`, `cohere` providers.** The `Provider` interface and resolution/ambiguity seam already
  covers all four; each remaining provider is a `newXProvider(cfg) (Provider, error)` following
  `embed/openai.go`/`embed/ollama.go`'s shape (hand-rolled JSON, `egress.Do`, `doWithRetry`).
  Cohere has existing vendored-client precedent elsewhere in the org to mirror
  (`pkg/plugin/processor/builtin/impl/cohere` in `ConduitIO/conduit`); Voyage has no existing
  Conduit precedent and needs its request/response shape sourced from Voyage's own API docs.
- **The chunking processor.** A separate, non-network processor (`Process(records) -> records`,
  fan-out 1 input record to N output chunk records) — no dependency on the egress capability, can
  land independently of the remaining providers.
- **Acceptance tests.** This slice has unit tests against a mocked `Provider`/`egress.HTTPService`
  seam; a `conduit-connector-sdk`-style acceptance suite (or an equivalent for processors) that
  exercises a resolved provider end-to-end is a follow-up, gated on having more than one working
  provider to prove the suite isn't OpenAI-shaped by accident.
- **The bundle end-to-end test.** Postgres CDC → chunk → embed → pgvector, CI-tested with records
  asserted at the vector store (design doc §8/Testing) — depends on the chunking processor and
  `conduit-connector-pgvector`, both out of scope here.
- **Metrics/observability wiring** (`conduit_embedding_tokens_total`,
  `conduit_embedding_call_duration_seconds`, design doc §9) — record-level `tokensUsed` metadata
  ships in this slice; the pipeline-level metrics counters are a host/engine-side follow-up.
