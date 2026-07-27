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

//go:build live_embed

// Live, API-key-gated smoke tests. These are NOT the acceptance gate — the
// mock-egress suite (acceptance_test.go) is. They are excluded from a normal
// build by the `live_embed` build tag AND additionally skip unless the
// relevant vendor key is present — two gates, so an ambient OPENAI_API_KEY (or
// similar) in a developer's or CI environment can never hijack a plain
// `go test ./...` into making a real, paid, network-dependent call. Run them
// explicitly with `go test -tags live_embed ./embed/...` and the keys set. No
// key ever appears in code or fixtures — each test reads it from the
// environment.
//
// Because provider auth is normally host-injected via AuthSecretRef (the raw
// key never enters guest memory), the live transport double below deliberately
// substitutes a real net/http client for the host's egress role: it reads the
// key from the environment and sets the real Authorization header itself. This
// bypasses the host egress policy on purpose, for a real-endpoint smoke test
// only.
package embed

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"testing"

	"github.com/conduitio/conduit-processor-sdk/egress"
	"github.com/conduitio/conduit-processor-sdk/pprocutils"
	"github.com/matryer/is"
)

// liveMaxBody caps the response body the live transport double will read, a
// smoke-test analogue of the host's real size cap.
const liveMaxBody = 8 << 20

// liveHTTPService is a real net/http-backed [pprocutils.HTTPService] used only
// by the live tier. It stands in for the host's egress role: it reads an API
// key (already resolved from the environment by the caller) and injects it as
// the Authorization header the real vendor endpoint expects — the one place in
// this repo a real credential ever reaches a real socket.
type liveHTTPService struct {
	apiKey string
}

func (s liveHTTPService) Do(ctx context.Context, req pprocutils.HTTPRequest) (pprocutils.HTTPResponse, error) {
	httpReq, err := http.NewRequestWithContext(ctx, req.Method, req.URL, bytes.NewReader(req.Body))
	if err != nil {
		return pprocutils.HTTPResponse{}, fmt.Errorf("build live request: %w", err)
	}
	for k, vs := range req.Headers {
		for _, v := range vs {
			httpReq.Header.Add(k, v)
		}
	}
	if s.apiKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+s.apiKey)
	}

	resp, err := http.DefaultClient.Do(httpReq)
	if err != nil {
		return pprocutils.HTTPResponse{}, fmt.Errorf("live request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(io.LimitReader(resp.Body, liveMaxBody))
	if err != nil {
		return pprocutils.HTTPResponse{}, fmt.Errorf("read live response body: %w", err)
	}
	return pprocutils.HTTPResponse{
		StatusCode: resp.StatusCode,
		Headers:    resp.Header,
		Body:       body,
	}, nil
}

// runLiveEmbed builds provider under a real transport keyed by apiKey, embeds
// two inputs, and asserts a plausible smoke result (two non-empty,
// equal-length vectors). Never logs the key or the request body.
func runLiveEmbed(t *testing.T, provider Provider, apiKey string) {
	t.Helper()
	is := is.New(t)

	egress.HTTPService = liveHTTPService{apiKey: apiKey}
	t.Cleanup(func() { egress.HTTPService = disabledService{} })

	result, err := provider.Embed(context.Background(), []string{"the quick brown fox", "jumps over the lazy dog"})
	is.NoErr(err)
	is.Equal(len(result.Outcomes), 2)
	is.NoErr(result.Outcomes[0].Err)
	is.NoErr(result.Outcomes[1].Err)
	is.True(len(result.Outcomes[0].Vector) > 0)
	is.Equal(len(result.Outcomes[0].Vector), len(result.Outcomes[1].Vector))
	is.Equal(result.Dimension, len(result.Outcomes[0].Vector))
}

func TestLive_OpenAI(t *testing.T) {
	key := os.Getenv("OPENAI_API_KEY")
	if key == "" {
		t.Skip("live: set OPENAI_API_KEY to run the OpenAI smoke test")
	}
	is := is.New(t)
	p, err := newOpenAIProvider(testOpenAIConfig())
	is.NoErr(err)
	runLiveEmbed(t, p, key)
}

func TestLive_Voyage(t *testing.T) {
	key := os.Getenv("VOYAGE_API_KEY")
	if key == "" {
		t.Skip("live: set VOYAGE_API_KEY to run the Voyage smoke test")
	}
	is := is.New(t)
	p, err := newVoyageProvider(testVoyageConfig())
	is.NoErr(err)
	runLiveEmbed(t, p, key)
}

func TestLive_Cohere(t *testing.T) {
	key := os.Getenv("COHERE_API_KEY")
	if key == "" {
		t.Skip("live: set COHERE_API_KEY to run the Cohere smoke test")
	}
	is := is.New(t)
	p, err := newCohereProvider(testCohereConfig())
	is.NoErr(err)
	runLiveEmbed(t, p, key)
}

func TestLive_Ollama(t *testing.T) {
	baseURL := os.Getenv("OLLAMA_BASE_URL")
	if baseURL == "" {
		t.Skip("live: set OLLAMA_BASE_URL to run the Ollama smoke test")
	}
	is := is.New(t)
	cfg := testOllamaConfig()
	cfg.OllamaBaseURL = baseURL
	p, err := newOllamaProvider(cfg)
	is.NoErr(err)

	// Ollama takes no API key (empty apiKey → no Authorization header) and
	// embeds one input per call, so drive a single-input call here rather than
	// runLiveEmbed's 2-input batch (which the provider rejects).
	egress.HTTPService = liveHTTPService{}
	t.Cleanup(func() { egress.HTTPService = disabledService{} })

	result, err := p.Embed(context.Background(), []string{"the quick brown fox"})
	is.NoErr(err)
	is.Equal(len(result.Outcomes), 1)
	is.True(len(result.Outcomes[0].Vector) > 0)
}
