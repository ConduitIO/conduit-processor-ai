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

package version

import "testing"

// TestUnstampedBuildIsExemptSentinel pins the load-bearing property that an
// ordinary build (no -ldflags -X) reports exactly conduit's dev-build sentinel.
// If this ever drifts to some other concrete string, an unstamped local build
// would stop being exempt from conduit's install-time version-equality check
// (validateProcessorWASM) and would fail to install against any index version —
// a silent, confusing break. The publish workflow is the ONLY thing that
// overwrites Value; nothing in code may.
func TestUnstampedBuildIsExemptSentinel(t *testing.T) {
	if Value != DevelSentinel {
		t.Fatalf("unstamped Value = %q, want the exempt dev-build sentinel %q; "+
			"only the publish workflow's -ldflags -X may set a concrete version", Value, DevelSentinel)
	}
	if DevelSentinel != "(devel)" {
		t.Fatalf("DevelSentinel = %q, want %q (Go's module-version default, which conduit's validateProcessorWASM exempts)", DevelSentinel, "(devel)")
	}
}
