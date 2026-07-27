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
	"errors"
	"testing"

	"github.com/matryer/is"
)

func noEnv(string) string { return "" }

func envMap(m map[string]string) func(string) string {
	return func(k string) string { return m[k] }
}

func TestResolveProviderName_ExplicitConfigWins(t *testing.T) {
	is := is.New(t)
	cfg := Config{Provider: "voyage", OpenAIAuthSecretRef: "openai-key"}
	name, err := ResolveProviderName(cfg, envMap(map[string]string{EnvProvider: "cohere"}))
	is.NoErr(err)
	is.Equal(name, "voyage")
}

func TestResolveProviderName_EnvWinsOverAutoDetect(t *testing.T) {
	is := is.New(t)
	cfg := Config{OpenAIAuthSecretRef: "openai-key"}
	name, err := ResolveProviderName(cfg, envMap(map[string]string{EnvProvider: "cohere"}))
	is.NoErr(err)
	is.Equal(name, "cohere")
}

func TestResolveProviderName_AutoDetectExactlyOne(t *testing.T) {
	is := is.New(t)
	cfg := Config{OpenAIAuthSecretRef: "openai-key"}
	name, err := ResolveProviderName(cfg, noEnv)
	is.NoErr(err)
	is.Equal(name, ProviderOpenAI)
}

func TestResolveProviderName_ZeroCandidatesRefused(t *testing.T) {
	is := is.New(t)
	_, err := ResolveProviderName(Config{}, noEnv)
	is.True(err != nil)
	var perr *Error
	is.True(errors.As(err, &perr))
	is.Equal(perr.Code, CodeNoProviderConfigured)
}

func TestResolveProviderName_AmbiguousCandidatesRefused(t *testing.T) {
	is := is.New(t)
	cfg := Config{OpenAIAuthSecretRef: "openai-key", CohereAuthSecretRef: "cohere-key"}
	_, err := ResolveProviderName(cfg, noEnv)
	is.True(err != nil)
	var perr *Error
	is.True(errors.As(err, &perr))
	is.Equal(perr.Code, CodeAmbiguousProvider)
}

func TestResolveProviderName_AmbiguousIncludesOllamaByBaseURL(t *testing.T) {
	is := is.New(t)
	cfg := Config{OpenAIAuthSecretRef: "openai-key", OllamaBaseURL: "http://localhost:11434"}
	_, err := ResolveProviderName(cfg, noEnv)
	var perr *Error
	is.True(errors.As(err, &perr))
	is.Equal(perr.Code, CodeAmbiguousProvider)
}

func TestBuildProvider_OpenAIImplemented(t *testing.T) {
	is := is.New(t)
	p, err := BuildProvider(ProviderOpenAI, Config{
		OpenAIAuthSecretRef: "openai-key",
		OpenAIBaseURL:       "https://api.openai.com",
		Model:               "text-embedding-3-small",
	})
	is.NoErr(err)
	is.Equal(p.Name(), ProviderOpenAI)
}

func TestBuildProvider_UnimplementedProvidersReturnCodedError(t *testing.T) {
	for _, name := range []string{ProviderVoyage, ProviderCohere, ProviderOllama} {
		t.Run(name, func(t *testing.T) {
			is := is.New(t)
			_, err := BuildProvider(name, Config{})
			is.True(err != nil)
			var perr *Error
			is.True(errors.As(err, &perr))
			is.Equal(perr.Code, CodeProviderNotImplemented)
		})
	}
}

func TestBuildProvider_UnknownProviderRefused(t *testing.T) {
	is := is.New(t)
	_, err := BuildProvider("does-not-exist", Config{})
	is.True(err != nil)
	var perr *Error
	is.True(errors.As(err, &perr))
	is.Equal(perr.Code, CodeInvalidConfig)
}

func TestIsImplemented(t *testing.T) {
	is := is.New(t)
	is.True(IsImplemented(ProviderOpenAI))
	is.True(!IsImplemented(ProviderVoyage))
	is.True(!IsImplemented(ProviderCohere))
	is.True(!IsImplemented(ProviderOllama))
}
