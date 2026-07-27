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

// This file is the embedding processor's in-repo acceptance suite. The
// conduit-processor-sdk has no generic processor acceptance harness (unlike
// conduit-connector-sdk's AcceptanceTest(t, driver)); this is a
// processor-specific contract suite that stands in for one — a declared
// compatibility contract authored for this processor.
//
// It has two tiers:
//
//   - The MOCK tier (this file's TestAcceptance_*) is the gate: it drives the
//     full Configure → Open → Process path for every provider against a mock
//     egress returning canned vendor JSON, plus a provider-independent
//     record-shape matrix. It runs with no network under `go test -race`,
//     matching CI.
//   - The LIVE tier (live_test.go) is a non-gating smoke test that hits the
//     real vendor endpoints only when an API key is present in the
//     environment, and skips otherwise.
package embed

import (
	"context"
	"errors"
	"net/http"
	"testing"

	"github.com/conduitio/conduit-commons/config"
	"github.com/conduitio/conduit-commons/opencdc"
	sdk "github.com/conduitio/conduit-processor-sdk"
	"github.com/conduitio/conduit-processor-sdk/pprocutils"
	"github.com/conduitio/conduit-processor-sdk/pprocutils/mock"
	"github.com/matryer/is"
	"go.uber.org/mock/gomock"
)

// providerAcceptanceCase declares one provider's end-to-end contract against a
// mock egress: the request it must emit (method/URL/auth) and the output the
// canned success body must map to (metadata + token accounting).
type providerAcceptanceCase struct {
	name        string
	cfg         config.Config
	body        []byte // canned vendor success body for a single-input request
	wantURL     string
	wantAuthRef string // "" for ollama, which takes no API key
	wantModel   string // ai.embedding.model value
	wantTokens  string // ai.embedding.tokensUsed value, "" when suppressed
	wantScope   string // ai.embedding.tokensUsedScope value, "" when suppressed
}

func providerAcceptanceCases(t *testing.T) []providerAcceptanceCase {
	t.Helper()
	const dim = 3
	return []providerAcceptanceCase{
		{
			name: ProviderOpenAI,
			cfg: config.Config{
				"openai.authSecretRef": "openai-key",
				"openai.baseURL":       "https://api.openai.com",
				"model":                "text-embedding-3-small",
			},
			body:        openAISuccessBody(t, map[int]int{0: dim}, 7),
			wantURL:     "https://api.openai.com/v1/embeddings",
			wantAuthRef: "openai-key",
			wantModel:   "text-embedding-3-small",
			wantTokens:  "7",
			wantScope:   "record",
		},
		{
			name: ProviderVoyage,
			cfg: config.Config{
				"voyage.authSecretRef": "voyage-key",
				"voyage.baseURL":       "https://api.voyageai.com",
				"model":                "voyage-3.5",
			},
			body:        voyageSuccessBody(t, map[int]int{0: dim}, 7),
			wantURL:     "https://api.voyageai.com/v1/embeddings",
			wantAuthRef: "voyage-key",
			wantModel:   "voyage-3.5",
			wantTokens:  "7",
			wantScope:   "record",
		},
		{
			name: ProviderCohere,
			cfg: config.Config{
				"cohere.authSecretRef": "cohere-key",
				"cohere.baseURL":       "https://api.cohere.com",
				"model":                "embed-english-v3.0",
			},
			body:        cohereSuccessBody(t, 1, dim, 7),
			wantURL:     "https://api.cohere.com/v1/embed",
			wantAuthRef: "cohere-key",
			wantModel:   "embed-english-v3.0", // set from config; Cohere doesn't echo it
			wantTokens:  "7",
			wantScope:   "record",
		},
		{
			name: ProviderOllama,
			cfg: config.Config{
				"ollama.baseURL": "http://127.0.0.1:11434",
				"model":          "nomic-embed-text",
			},
			body:        ollamaSuccessBody(t, dim),
			wantURL:     "http://127.0.0.1:11434/api/embeddings",
			wantAuthRef: "", // local Ollama takes no API key
			wantModel:   "nomic-embed-text",
			wantTokens:  "", // Ollama reports no usage → TokensScopeUnknown → suppressed
			wantScope:   "",
		},
	}
}

// TestAcceptance_ProviderSeamMatrix is the provider seam contract: each real
// provider driven end-to-end (Configure → Open → Process) against a mock
// egress, asserting request round-trip and response→record mapping. A single
// input keeps each provider to exactly one egress call (Ollama caps at one
// input per call); multi-input sub-batching and batch-scoped tokens are
// covered per-provider in openai_test.go / voyage_test.go / cohere_test.go.
func TestAcceptance_ProviderSeamMatrix(t *testing.T) {
	for _, tc := range providerAcceptanceCases(t) {
		t.Run(tc.name, func(t *testing.T) {
			is := is.New(t)
			ctrl := gomock.NewController(t)
			svc := mock.NewHTTPService(ctrl)
			withHTTPService(t, svc)

			var captured pprocutils.HTTPRequest
			svc.EXPECT().Do(gomock.Any(), gomock.Any()).DoAndReturn(
				func(_ context.Context, req pprocutils.HTTPRequest) (pprocutils.HTTPResponse, error) {
					captured = req
					return pprocutils.HTTPResponse{StatusCode: http.StatusOK, Body: tc.body}, nil
				}).Times(1)

			p := NewProcessor()
			p.getenv = func(string) string { return "" } // isolate from a stray CONDUIT_EMBED_PROVIDER
			is.NoErr(p.Configure(context.Background(), tc.cfg))
			is.NoErr(p.Open(context.Background()))

			out := p.Process(context.Background(), []opencdc.Record{recordWithText("hello")})
			is.Equal(len(out), 1)
			single, ok := out[0].(sdk.SingleRecord)
			is.True(ok) // happy path must be a SingleRecord, never an ErrorRecord
			rec := opencdc.Record(single)

			// Request round-trip: method, URL, AuthSecretRef set as expected,
			// and NEVER a raw Authorization header (the host injects it).
			is.Equal(captured.Method, http.MethodPost)
			is.Equal(captured.URL, tc.wantURL)
			is.Equal(captured.AuthSecretRef, tc.wantAuthRef)
			_, hasAuth := captured.Headers["Authorization"]
			is.True(!hasAuth)

			// Response→record mapping: native vector at the default output
			// field, and the metadata contract.
			after, ok := rec.Payload.After.(opencdc.StructuredData)
			is.True(ok)
			vec, ok := after["vector"].([]any)
			is.True(ok) // native []any, never a JSON-encoded []byte
			is.Equal(len(vec), 3)
			is.Equal(rec.Metadata[MetadataProvider], tc.name)
			is.Equal(rec.Metadata[MetadataModel], tc.wantModel)
			is.Equal(rec.Metadata[MetadataDimension], "3")

			// Token accounting: present only when the provider reported usage.
			gotTokens, hasTokens := rec.Metadata[MetadataTokensUsed]
			gotScope := rec.Metadata[MetadataTokensScope]
			if tc.wantScope == "" {
				is.True(!hasTokens) // scope unknown → tokensUsed metadata suppressed, never a fabricated 0
				is.Equal(gotScope, "")
			} else {
				is.Equal(gotTokens, tc.wantTokens)
				is.Equal(gotScope, tc.wantScope)
			}
		})
	}
}

// recordShapeCase declares one input record shape and the ProcessedRecord
// variant (and, on failure, the coded error) the processor must produce for
// it — provider-independent, so it drives a fakeProvider seam.
type recordShapeCase struct {
	name        string
	cfg         map[string]string
	record      opencdc.Record
	provider    Provider
	wantErrCode string // "" ⇒ expect a SingleRecord
	wantEmbed   bool   // whether the record should carry a vector afterwards
}

func TestAcceptance_RecordShapeMatrix(t *testing.T) {
	okProvider := func() Provider {
		return &fakeProvider{name: "fake", maxBatch: 96, resultFn: allSucceed(3, 5)}
	}
	failProvider := func() Provider {
		return &fakeProvider{
			name: "fake", maxBatch: 96,
			resultFn: func([]string) (BatchResult, error) {
				return BatchResult{}, errors.New("provider unreachable")
			},
		}
	}

	cases := []recordShapeCase{
		{
			name:      "structured_payload",
			record:    recordWithText("embed me"),
			provider:  okProvider(),
			wantEmbed: true,
		},
		{
			// Raw-bytes input: reads the record's RawData Key (an opencdc.Data,
			// exercising resolveInputText's []byte→string path) and writes the
			// vector to a structured payload field.
			name: "raw_bytes_input_field",
			cfg:  map[string]string{"inputField": ".Key"},
			record: opencdc.Record{
				Key:      opencdc.RawData("raw text to embed"),
				Metadata: opencdc.Metadata{},
				Payload:  opencdc.Change{After: opencdc.StructuredData{}},
			},
			provider:  okProvider(),
			wantEmbed: true,
		},
		{
			name: "tombstone_passes_through_unembedded",
			record: opencdc.Record{
				Operation: opencdc.OperationDelete,
				Key:       opencdc.RawData("orders:42"),
				Metadata:  opencdc.Metadata{"ai.chunk.source_key": "orders:42"},
			},
			provider:  okProvider(),
			wantEmbed: false, // a delete passes through unchanged, never embedded
		},
		{
			name:        "unresolvable_input_field",
			record:      opencdc.Record{Metadata: opencdc.Metadata{}}, // Payload.After is nil
			provider:    okProvider(),
			wantErrCode: CodeFieldResolution,
		},
		{
			name:        "provider_failure",
			record:      recordWithText("embed me"),
			provider:    failProvider(),
			wantErrCode: CodeProviderError,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			is := is.New(t)
			p := newTestProcessor(t, tc.cfg, tc.provider)

			out := p.Process(context.Background(), []opencdc.Record{tc.record})
			is.Equal(len(out), 1)

			if tc.wantErrCode != "" {
				errRec, ok := out[0].(sdk.ErrorRecord)
				is.True(ok) // failure shapes must be ErrorRecord, never a silent SingleRecord
				var perr *Error
				is.True(errors.As(errRec.Error, &perr))
				is.Equal(perr.Code, tc.wantErrCode)
				is.True(perr.Suggestion != "") // errors are API: every failure carries a fix hint
				return
			}

			single, ok := out[0].(sdk.SingleRecord)
			is.True(ok)
			rec := opencdc.Record(single)
			if tc.wantEmbed {
				is.Equal(rec.Metadata[MetadataProvider], "fake") // embedded ⇒ metadata attached
			} else {
				// Passthrough (tombstone): no embedding metadata, original op intact.
				_, hasProvider := rec.Metadata[MetadataProvider]
				is.True(!hasProvider)
				is.Equal(rec.Operation, opencdc.OperationDelete)
			}
		})
	}
}

// TestAcceptance_ProviderFailurePathMatrix runs each real provider through a
// failure path (a 400 the vendor rejects), proving the coded-error contract
// holds uniformly across providers via the mock egress.
func TestAcceptance_ProviderFailurePathMatrix(t *testing.T) {
	for _, tc := range providerAcceptanceCases(t) {
		t.Run(tc.name, func(t *testing.T) {
			is := is.New(t)
			ctrl := gomock.NewController(t)
			svc := mock.NewHTTPService(ctrl)
			withHTTPService(t, svc)

			svc.EXPECT().Do(gomock.Any(), gomock.Any()).Return(pprocutils.HTTPResponse{
				StatusCode: http.StatusBadRequest,
				Body:       []byte(`{"message":"bad","detail":"bad","error":"bad"}`),
			}, nil).Times(1) // 400 is not retried by any provider

			p := NewProcessor()
			p.getenv = func(string) string { return "" }
			is.NoErr(p.Configure(context.Background(), tc.cfg))
			is.NoErr(p.Open(context.Background()))

			out := p.Process(context.Background(), []opencdc.Record{recordWithText("hello")})
			is.Equal(len(out), 1)
			errRec, ok := out[0].(sdk.ErrorRecord)
			is.True(ok) // a provider 400 fails the record with a coded error, never a silent pass-through
			var perr *Error
			is.True(errors.As(errRec.Error, &perr))
			is.Equal(perr.Code, CodeProviderError)
			is.True(perr.Suggestion != "")
		})
	}
}
