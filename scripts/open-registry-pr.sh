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

# publish.yml's register step. Turns the build step's `artifacts` output into two
# per-processor registry source files (index/processors/ai.chunk.json and
# index/processors/ai.embed.json), each a serialized index.Processor with ONE
# arch-neutral (wasip1/wasm/wasm-processor) artifact, then opens/updates a single
# PR against the index repo. index-sign.yml (role=root) later assembles these into
# the signed, served index; this script NEVER root-signs and NEVER merges.
#
# Safety properties (mirroring connector-publish-action's routing.Decide, in
# plain reviewable shell because that action's Go binary models connectors[] only
# and cannot be reused for processors — see OQ-C in the bootstrap plan):
#
#   * append-only: for a name that already exists in the index, the new version is
#     PREPENDED to versions[] (newest-first); if that exact version is already
#     present the file is left untouched (idempotent re-run), never rewritten.
#   * identity is set ONLY on first registration: for an existing name the
#     publisher block is copied verbatim from the current file and never changed
#     here — changing a pinned identity is a human-reviewed registry-side action,
#     not something a publish run can do.
#   * first registration is human-gated: the PR is opened and labeled, never
#     merged. The single point where typosquatting has no cryptographic backstop
#     (conduit's findProcessor is exact-match, no fuzzy suggestion) is the reviewer.
#
# Requires: jq, git, gh. GH_TOKEN must be a token scoped to open/update PRs on
# INDEX_REPO only (no merge, no admin) — this repo's blast radius if leaked: it
# can propose, never land.
set -euo pipefail

: "${ARTIFACTS:?ARTIFACTS (build step JSON output) is required}"
: "${VERSION_TAG:?}"
# RFC3339 publish timestamp. Generated in-runner (not taken from event payload):
# a tag push has no reliable head_commit, and event fields are untrusted input.
RELEASED_AT="${RELEASED_AT:-$(date -u +%Y-%m-%dT%H:%M:%SZ)}"
: "${MIN_CONDUIT_VERSION:?}"
: "${MIN_PROTOCOL_VERSION:?}"
: "${PROVENANCE_BUNDLE_URL:?}"

# Read the predicateType out of the attestation we actually published rather
# than hardcoding one. The v0.1.0 entry initially claimed
# https://slsa.dev/provenance/v1 while slsa-github-generator had emitted v0.2 —
# harmless today (the installer reads the statement's own predicateType, not the
# index's copy) but a signed index must not assert something false, and any
# constant here goes stale the moment the generator bumps its predicate.
PREDICATE_TYPE="$(
  curl -sSL --fail --max-time 60 "$PROVENANCE_BUNDLE_URL" |
    head -n 1 |
    jq -r '(.dsseEnvelope.payload // .payload)' |
    base64 -d |
    jq -r '.predicateType'
)"
case "$PREDICATE_TYPE" in
  https://slsa.dev/provenance/v*) ;;
  *)
    echo "::error::could not read a SLSA predicateType from ${PROVENANCE_BUNDLE_URL} (got '${PREDICATE_TYPE}')" >&2
    exit 1
    ;;
esac
: "${INDEX_REPO:?}"
: "${GH_TOKEN:?GH_TOKEN (index-repo-token) is required to open a PR}"
: "${GITHUB_REPOSITORY:?}"
: "${REPO_URL:?REPO_URL (source repository html url) is required}"
: "${FIRST_REG_IDENTITY_PATTERN:?first-registration identity ref-pattern fragment is required}"

SEMVER="${VERSION_TAG#v}"

# Per-name web-UI metadata (not trust-bearing; written on first registration only).
displayName_for() {
  case "$1" in
    ai.chunk) echo "Conduit AI Chunking Processor" ;;
    ai.embed) echo "Conduit AI Embedding Processor" ;;
    *)        echo "$1" ;;
  esac
}
description_for() {
  case "$1" in
    ai.chunk) echo "Split a record's text into chunk records using a configurable strategy (standalone WASM)." ;;
    ai.embed) echo "Generate vector embeddings for records using a pluggable provider (standalone WASM)." ;;
    *)        echo "Standalone WASM processor from conduit-processor-ai." ;;
  esac
}

# Build the fully-anchored expectedIdentityPattern for a FIRST registration:
#   ^https://github\.com/<repo>/\.github/workflows/publish\.yml@<ref-pattern>$
# Regex-escape the literal dots in the repo path; the ref-pattern fragment is
# supplied already-escaped by the workflow (a maintainer consciously scopes trust
# to tags — never auto-derived from the current ref).
ESC_REPO="$(printf '%s' "$GITHUB_REPOSITORY" | sed 's/\./\\./g')"
IDENTITY_PATTERN="^https://github\\.com/${ESC_REPO}/\\.github/workflows/publish\\.yml@${FIRST_REG_IDENTITY_PATTERN}\$"

CLONE_DIR="$(mktemp -d)"
git clone --depth 1 "https://x-access-token:${GH_TOKEN}@github.com/${INDEX_REPO}.git" "$CLONE_DIR"
cd "$CLONE_DIR"
git config user.name "conduit-processor-ai-publish"
git config user.email "actions@users.noreply.github.com"

MARKER="<!-- conduit-processor-ai-publish:${VERSION_TAG} -->"
BRANCH="processor-ai-publish/${SEMVER}"

EXISTING_PR_NUMBER="$(gh pr list --repo "$INDEX_REPO" --state open --search "$MARKER in:body" --json number --jq '.[0].number // empty')"
if [ -n "$EXISTING_PR_NUMBER" ]; then
  EXISTING_BRANCH="$(gh pr view "$EXISTING_PR_NUMBER" --repo "$INDEX_REPO" --json headRefName --jq '.headRefName')"
  git fetch origin "$EXISTING_BRANCH"
  git checkout "$EXISTING_BRANCH"
  BRANCH="$EXISTING_BRANCH"
else
  git checkout -b "$BRANCH"
fi

NEW_NAME_REGISTRATION="false"
CHANGED_FILES=()

COUNT="$(echo "$ARTIFACTS" | jq 'length')"
for i in $(seq 0 $((COUNT - 1))); do
  NAME="$(echo "$ARTIFACTS" | jq -r ".[$i].name")"
  URL="$(echo "$ARTIFACTS" | jq -r ".[$i].url")"
  SHA="$(echo "$ARTIFACTS" | jq -r ".[$i].sha256")"
  SIZE="$(echo "$ARTIFACTS" | jq -r ".[$i].size")"
  SIGURL="$(echo "$ARTIFACTS" | jq -r ".[$i].signatureBundleURL")"
  FILE="index/processors/${NAME}.json"

  NEW_VERSION="$(jq -n \
    --arg version "$SEMVER" --arg releasedAt "$RELEASED_AT" \
    --arg minc "$MIN_CONDUIT_VERSION" --arg minp "$MIN_PROTOCOL_VERSION" \
    --arg url "$URL" --arg sha "$SHA" --argjson size "$SIZE" \
    --arg sigurl "$SIGURL" --arg provurl "$PROVENANCE_BUNDLE_URL" \
    --arg predtype "$PREDICATE_TYPE" \
    '{
      version: $version, releasedAt: $releasedAt,
      minConduitVersion: $minc, minProtocolVersion: $minp,
      artifact: {
        os: "wasip1", arch: "wasm", kind: "wasm-processor",
        url: $url, sha256: $sha, size: $size,
        signature: { bundleURL: $sigurl }
      },
      slsaProvenance: { bundleURL: $provurl, predicateType: $predtype },
      deprecated: false
    }')"

  # A hand-authored bootstrap entry carries one placeholder version whose
  # artifact digest is all zeros — the "not yet published" sentinel. The first
  # real publish of that version REPLACES the sentinel (so the real digest lands
  # and no placeholder cruft remains); every OTHER already-present version is
  # immutable (append-only — a re-run of an already-published tag is a no-op).
  ZERO_SHA="0000000000000000000000000000000000000000000000000000000000000000"

  if [ -f "$FILE" ]; then
    EXISTING_DIGEST="$(jq -r --arg v "$SEMVER" 'first(.versions[] | select(.version == $v) | .artifact.sha256) // ""' "$FILE")"
    if [ -n "$EXISTING_DIGEST" ] && [ "$EXISTING_DIGEST" != "$ZERO_SHA" ]; then
      echo "::notice::${NAME} version ${SEMVER} already published in ${FILE} — leaving untouched (append-only, idempotent)" >&2
      continue
    fi
    # Prepend the new version, dropping any sentinel placeholder of the SAME
    # version; publisher/name/metadata are preserved verbatim (identity is never
    # rewritten by an automated publish).
    TMP="$(mktemp)"
    jq --arg v "$SEMVER" --argjson new "$NEW_VERSION" \
      '.versions = ([$new] + (.versions | map(select(.version != $v))))' "$FILE" >"$TMP"
    mv "$TMP" "$FILE"
  else
    NEW_NAME_REGISTRATION="true"
    jq -n \
      --arg name "$NAME" \
      --arg displayName "$(displayName_for "$NAME")" \
      --arg description "$(description_for "$NAME")" \
      --arg repository "$REPO_URL" \
      --arg issuer "https://token.actions.githubusercontent.com" \
      --arg identity "$IDENTITY_PATTERN" \
      --argjson version "$NEW_VERSION" \
      '{
        name: $name, displayName: $displayName, description: $description,
        repository: $repository,
        publisher: { expectedOIDCIssuer: $issuer, expectedIdentityPattern: $identity },
        versions: [ $version ]
      }' >"$FILE"
  fi
  git add "$FILE"
  CHANGED_FILES+=("$FILE")
done

if git diff --cached --quiet; then
  echo "::notice::no index changes to commit (all entries already up to date for ${VERSION_TAG})" >&2
  exit 0
fi

PR_TITLE="feat(index): conduit-processor-ai ${VERSION_TAG} (ai.chunk, ai.embed)"
PR_BODY_FILE="$(mktemp)"
{
  echo "$MARKER"
  echo
  echo "Automated publish of \`conduit-processor-ai\` ${VERSION_TAG}."
  echo
  echo "- Files: ${CHANGED_FILES[*]}"
  echo "- Artifacts are cosign-keyless-signed as this repo's \`publish.yml@${VERSION_TAG}\` identity, with SLSA provenance."
  echo "- Digests/sizes/URLs are the publish job's real outputs, computed from the archives as written."
  echo
  if [ "$NEW_NAME_REGISTRATION" = "true" ]; then
    echo "**First registration of a new processor name — root-of-trust decision.**"
    echo "Reviewer MUST verify (anti-typosquat, no cryptographic backstop on first registration):"
    echo "- \`name\` is the intended install id (\`ai.chunk\` / \`ai.embed\`), not a look-alike."
    echo "- \`publisher.expectedIdentityPattern\` pins THIS repo's \`publish.yml\` at a version tag."
    echo "- \`minConduitVersion\` is the version that ships \`processor-plugins install\`."
    echo
    echo "This PR is NOT auto-merged."
  else
    echo "Append-only version bump for an already-registered name; publisher identity unchanged."
  fi
} >"$PR_BODY_FILE"

git commit -m "$PR_TITLE"
git push origin "$BRANCH"

if [ -n "$EXISTING_PR_NUMBER" ]; then
  PR_URL="$(gh pr view "$EXISTING_PR_NUMBER" --repo "$INDEX_REPO" --json url --jq '.url')"
  echo "::notice::updated existing index PR: $PR_URL" >&2
else
  LABEL_ARGS=()
  if [ "$NEW_NAME_REGISTRATION" = "true" ]; then
    # Create the label idempotently before using it. gh pr create FAILS outright
    # on an unknown label, which would abort the publish at its very last step —
    # after the artifacts are built, signed and uploaded — leaving a released
    # version with no index entry and a human to reconcile it by hand. A fresh
    # or renamed registry repo should self-heal instead of ambushing the
    # release. `|| true` because the only expected failure is "already exists".
    gh label create "new-processor-registration" \
      --repo "$INDEX_REPO" \
      --description "Adds a processor name not previously in the index (needs naming review)" \
      --color "0E8A16" \
      --force >/dev/null 2>&1 || true
    LABEL_ARGS+=(--label "new-processor-registration")
  fi
  PR_URL="$(gh pr create --repo "$INDEX_REPO" --title "$PR_TITLE" --body-file "$PR_BODY_FILE" --head "$BRANCH" "${LABEL_ARGS[@]}")"
  echo "::notice::opened index PR: $PR_URL" >&2
fi
echo "pr-url=$PR_URL" >>"${GITHUB_OUTPUT:-/dev/null}"
