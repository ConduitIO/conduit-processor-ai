# conduit-processor-ai

Conduit processors for AI pipelines: chunking + embedding with pluggable providers (OpenAI,
Voyage, Ollama, Cohere). Part of Conduit v0.20 (WS8) —
see `docs/design-documents/20260724-ai-pipeline-components.md` in `ConduitIO/conduit` for the
full design.

## Status: Slice 1 (`ai.embed`, OpenAI only)

This is the **first reviewable slice** of the embedding processor. Scope:

- The `ai.embed` standalone-WASM processor, using `conduit-processor-sdk`'s host-mediated
  network-egress capability (`egress.Do`) for outbound HTTP — a WASI Preview 1 guest has no
  socket API of its own.
- One working provider: **OpenAI** (`POST /v1/embeddings`).
- The `Provider` seam (config, resolution, ambiguity detection) is wired for **all four**
  providers the design doc names (OpenAI, Voyage, Cohere, Ollama), but only OpenAI has a working
  implementation — selecting the other three yields a coded
  `ai.embedding_provider_not_implemented` error. See "Slicing note" below.
- The **chunking processor** is not part of this slice.

## Why this hand-rolls JSON instead of reusing a vendor SDK

The design doc's provider table frames OpenAI/Cohere's zero-new-dependency justification around
reusing the already-vendored `go-openai`/`cohere-go` clients the core engine's built-in
`openai.embeddings`/`cohere.embed` processors use. That justification does not carry over here:
this processor is a **standalone WASM guest** (WASI Preview 1, no socket API — the entire reason
the host-egress capability exists), and every vendor SDK does its own `net/http` dialing
internally, which cannot run inside the guest sandbox. The only coherent implementation is:
hand-roll each provider's request/response JSON as thin structs (the endpoints are simple
JSON-in/JSON-out) and call `egress.Do` for the actual transport, which the host performs under
its allowlist/DNS-rebinding/timeout/size-cap policy. See `embed/openai.go` and `embed/doc.go`.

## Delivery semantics — read before wiring this into a pipeline

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
  `tokensUsedScope: record`, which is exact for that one record.

## Configuration reference

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `provider` | string | _(empty)_ | Explicit provider: `openai` \| `voyage` \| `cohere` \| `ollama`. See Resolution below. |
| `model` | string | _(empty, required once resolved)_ | Provider embedding model, e.g. `text-embedding-3-small`. |
| `inputField` | string | `.Payload.After` | Record field read as the text to embed. |
| `outputField` | string | `.Payload.After` | Record field the embedding vector (JSON array) is written to. Set explicitly (e.g. `.Payload.After.embedding`) to preserve the original text in a structured record. |
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
| `ollama.baseURL` | string | _(empty)_ | Wired for resolution/ambiguity detection; `ollama` provider not yet implemented. |

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

### Output

On success, each record gets:

- The embedding vector as a JSON `[]float32` array at `outputField`.
- Metadata: `ai.embedding.provider`, `ai.embedding.model`, `ai.embedding.dimension`,
  `ai.embedding.tokensUsed` (only when the provider reported usage — see the tokensUsed note
  above), `ai.embedding.tokensUsedScope` (`record` or `batch`).

## Example pipeline

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
          inputField: .Payload.After.text
          outputField: .Payload.After.embedding
          maxTextsPerBatch: "96"
```

## Building

```sh
# Host-arch build, for running tests:
go build ./...

# The real target — standalone WASM:
GOOS=wasip1 GOARCH=wasm go build -tags wasm -o embedding.wasm ./cmd/embedding

go vet ./...
go test -race ./...
```

## Development note: the `go.mod` replace directive

`go.mod` currently has a `replace` pointing at a local, unreleased checkout of
`conduit-processor-sdk`'s `feat/wasm-host-egress` branch — the `egress` package this repo depends
on isn't tagged yet. **This must be repointed to a tagged `conduit-processor-sdk` release before
this repo's PR merges.** A `.golangci.yml` exclusion scoped to `go.mod` documents this; remove
both the `replace` and the exclusion together once a tagged SDK release ships the `egress`
package.

## Slicing note — what's deliberately not in this slice

- **`voyage`, `cohere`, `ollama` providers.** The `Provider` interface and resolution/ambiguity
  seam already covers all four; each remaining provider is a `newXProvider(cfg) (Provider, error)`
  following `embed/openai.go`'s shape (hand-rolled JSON, `egress.Do`, `doWithRetry`). Cohere and
  Ollama have existing hand-rolled/vendored-client precedent elsewhere in the org to mirror
  (`pkg/plugin/processor/builtin/impl/cohere`, `.../ollama` in `ConduitIO/conduit`); Voyage has no
  existing Conduit precedent and needs its request/response shape sourced from Voyage's own API
  docs.
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
