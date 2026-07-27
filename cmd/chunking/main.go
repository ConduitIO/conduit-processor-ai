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

//go:build wasm

// Command chunking is the standalone-WASM entry point for the ai.chunk
// processor. Build with:
//
//	GOOS=wasip1 GOARCH=wasm go build -tags wasm -o chunking.wasm ./cmd/chunking
package main

import (
	"github.com/conduitio/conduit-processor-ai/chunk"
	sdk "github.com/conduitio/conduit-processor-sdk"
)

func main() {
	sdk.Run(chunk.NewProcessor())
}
