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

// Package version carries the single build-stamped version string reported by
// every processor in this module's Specification().Version.
//
// # Why this exists
//
// A standalone WASM processor is published to the Conduit registry as a
// name@version index entry, and conduit's install-time validateProcessorWASM
// gate compares the running module's Specification().Version against the
// resolved index version — refusing the install when a concrete guest version
// disagrees with the index (a mispublished artifact). The one documented
// exemption is the Go dev-build sentinel "(devel)": conduit treats it as an
// unversioned local build and skips the equality check.
//
// So this variable MUST default to Value == devel sentinel for ordinary
// `go build`/`go test`/local runs (which then install without a version match),
// and MUST be overwritten with the real release semver at publish time so the
// published .wasm carries the version it claims. The publish workflow
// (.github/workflows/publish.yml) does that with the linker:
//
//	go build -ldflags "-X github.com/conduitio/conduit-processor-ai/internal/version.Value=v1.2.3" ...
//
// The string follows the conduit-processor-sdk convention of a leading "v"
// (e.g. "v0.1.0"); the registry index entry carries the same semver WITHOUT the
// leading "v" (schema forbids it), and conduit's NormalizeVersion reconciles the
// two ("v0.1.0" and "0.1.0" compare equal). See validateProcessorWASM.
package version

// DevelSentinel is Go's own module-version default for a binary not built via
// `go install module@version`; conduit's validateProcessorWASM exempts a guest
// module reporting exactly this value from the index-version equality check.
// Kept as a named constant so the default below is self-documenting and a test
// can assert "an unstamped build is the exempt sentinel, not some other string".
const DevelSentinel = "(devel)"

// Value is the version string every processor in this module reports from
// Specification().Version. It defaults to DevelSentinel and is overwritten at
// link time by the publish workflow's -ldflags -X for a real release. Do not
// set this anywhere in code — the linker is the only writer.
var Value = DevelSentinel
