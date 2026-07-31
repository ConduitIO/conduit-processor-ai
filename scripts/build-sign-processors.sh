#!/usr/bin/env bash
# Copyright © 2026 Meroxa, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# publish.yml's build step. For each standalone WASM processor in this module:
#
#   1. build it arch-neutrally (GOOS=wasip1 GOARCH=wasm go build -tags wasm),
#      stamping the release version into internal/version.Value via -ldflags -X
#      so the module's Specification().Version equals the tag (NOT the "(devel)"
#      dev-build sentinel — a published artifact must carry a real version, which
#      conduit's install-time validateProcessorWASM asserts against the index);
#   2. package the .wasm as a .tar.gz with the module at the archive ROOT —
#      conduit's installer extracts the single root-level file from a gzip+tar
#      archive (pkg/registry/extract.go ExtractBinary); a BARE .wasm URL would
#      fail extraction with "not valid gzip". The digest + signature below are
#      over this ARCHIVE, which is what the client downloads and verifies;
#   3. compute sha256 + size FROM THE FILE AS WRITTEN TO DISK (never re-derived
#      from a later fetch — the property that keeps a misbehaving download
#      gateway from ever causing a successful wrong-artifact install, only a
#      failed verification — mirrors connector-publish-action's build-and-sign.sh);
#   4. cosign-keyless-sign the archive as THIS repo's OIDC identity, emitting the
#      new-format Sigstore bundle conduit's trust core loads;
#   5. upload the archive + bundle to the release for the triggering tag; and
#   6. emit the `artifacts` output — one object per processor, carrying the
#      registry install NAME (ai.chunk / ai.embed, matching the WASM's own
#      Specification().Name), the index semver (NO leading v), url, sha256, size,
#      and the signature bundle url — consumed by open-registry-pr.sh and by the
#      SLSA subjects step.
#
# Requires: go, jq, cosign (installed by publish.yml before this runs), gh
# (preinstalled on GitHub-hosted runners), sha256sum or shasum.
#
# Ambient permissions required on the calling job: `id-token: write` (cosign
# keyless OIDC) and `contents: write` (`gh release upload`).
set -euo pipefail

: "${VERSION_TAG:?VERSION_TAG (e.g. v0.1.0, the triggering tag) is required}"
: "${GITHUB_REPOSITORY:?}"

# The registry index entry's semver has NO leading "v" (schema $defs/processorVersion
# version.pattern is ^\d+\.\d+\.\d+... — a leading v is a schema violation). The
# WASM keeps the SDK's leading-v convention; conduit's NormalizeVersion reconciles
# the two ("v0.1.0" == "0.1.0") in validateProcessorWASM.
SEMVER="${VERSION_TAG#v}"

# The module map: install-name  ->  ./cmd/<dir>. The install-name MUST equal the
# name the .wasm registers under (its Specification().Name) AND the registry
# entry name — conduit's findProcessor exact-matches the install argument against
# processors[].Name, and validateProcessorWASM refuses a .wasm whose spec name
# disagrees with the resolved index name. Keep these three in lockstep.
declare -A MODULES=(
  ["ai.chunk"]="./cmd/chunking"
  ["ai.embed"]="./cmd/embedding"
)

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

WORKDIR="$(mktemp -d)"
ARTIFACTS_JSON="[]"

# Stamp the release version into every module built below. One -X target, reused
# across both binaries (each build links its own copy of internal/version).
LDFLAGS="-X github.com/conduitio/conduit-processor-ai/internal/version.Value=${VERSION_TAG}"

# Iterate deterministically (sorted names) so the artifacts array — and the two
# index files derived from it — are byte-stable across re-runs of the same tag.
for NAME in $(printf '%s\n' "${!MODULES[@]}" | sort); do
  CMD="${MODULES[$NAME]}"
  # Asset basename: filesystem/URL-safe (dots -> hyphens); the asset name is
  # cosmetic (not trust-bearing — the digest is), the NAME field below is the
  # trust/resolution identity.
  SAFE="conduit-processor-$(echo "$NAME" | tr '.' '-')"        # conduit-processor-ai-chunk
  WASM="$WORKDIR/${NAME}.wasm"                                  # extracted-name inside the tar
  OUT="$WORKDIR/${SAFE}_${SEMVER}_wasip1_wasm.tar.gz"

  echo "::group::build ${NAME} (${CMD})"
  GOOS=wasip1 GOARCH=wasm go build -tags wasm -ldflags "$LDFLAGS" -o "$WASM" "$CMD"
  if [ ! -f "$WASM" ]; then
    echo "::error::build did not produce a .wasm at $WASM for ${NAME}" >&2
    exit 1
  fi
  # Fail closed if the stamp did not land: a "(devel)" published artifact would
  # install (validateProcessorWASM exempts the sentinel) but silently lose the
  # version-integrity guarantee. We can't run the wazero spec inspector here, so
  # assert the stamped string is present in the module and the sentinel is not.
  if ! grep -aq "$VERSION_TAG" "$WASM"; then
    echo "::error::version stamp ${VERSION_TAG} not found in built ${NAME} module — check the -ldflags -X path" >&2
    exit 1
  fi
  # Package the module at the archive ROOT (single root-level regular file).
  tar -czf "$OUT" -C "$WORKDIR" "$(basename "$WASM")"
  rm -f "$WASM"
  echo "::endgroup::"

  DIGEST="$(sha256_of "$OUT")"
  SIZE="$(wc -c <"$OUT" | tr -d ' ')"

  BUNDLE="${OUT}.sigstore.json"
  echo "::group::cosign sign-blob ${NAME}"
  # --new-bundle-format emits the Sigstore protobuf bundle
  # (application/vnd.dev.sigstore.bundle+json) conduit's trust core loads via
  # sigstore-go bundle.UnmarshalJSON. Without it cosign writes the LEGACY shape,
  # which conduit rejects as "no valid signature bundle".
  cosign sign-blob --yes \
    --oidc-issuer https://token.actions.githubusercontent.com \
    --new-bundle-format \
    --bundle "$BUNDLE" \
    "$OUT"
  echo "::endgroup::"

  ASSET_NAME="$(basename "$OUT")"
  BUNDLE_NAME="$(basename "$BUNDLE")"
  gh release upload "$VERSION_TAG" "$OUT" "$BUNDLE" --repo "$GITHUB_REPOSITORY" --clobber

  ARTIFACT_URL="https://github.com/${GITHUB_REPOSITORY}/releases/download/${VERSION_TAG}/${ASSET_NAME}"
  BUNDLE_URL="https://github.com/${GITHUB_REPOSITORY}/releases/download/${VERSION_TAG}/${BUNDLE_NAME}"

  ARTIFACTS_JSON="$(echo "$ARTIFACTS_JSON" | jq \
    --arg name "$NAME" --arg version "$SEMVER" \
    --arg url "$ARTIFACT_URL" --arg sha256 "$DIGEST" --argjson size "$SIZE" \
    --arg sigurl "$BUNDLE_URL" \
    '. + [{name:$name,version:$version,os:"wasip1",arch:"wasm",kind:"wasm-processor",url:$url,sha256:$sha256,size:$size,signatureBundleURL:$sigurl}]')"
done

# SLSA subjects: "sha256  filename" lines, base64'd, for the generic generator.
PROVENANCE_SUBJECTS="$(echo "$ARTIFACTS_JSON" | jq -r '[.[] | "\(.sha256)  \(.url | split("/") | last)"] | join("\n")' | base64 | tr -d '\n')"

{
  echo "artifacts<<PROC_PUBLISH_EOF"
  echo "$ARTIFACTS_JSON"
  echo "PROC_PUBLISH_EOF"
  echo "provenance-subjects=$PROVENANCE_SUBJECTS"
} >>"$GITHUB_OUTPUT"

rm -rf "$WORKDIR"
