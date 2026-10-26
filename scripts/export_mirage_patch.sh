#!/usr/bin/env bash
# Export the local MIRAGE double-insert fix as a reproducible patch.
#
# Run once inside the Linux/WSL2 environment that holds the fixed upstream
# tree (the 2026-10-22 validation used /root/mirage at 6749e9c):
#
#   bash scripts/export_mirage_patch.sh [/path/to/mirage]
#
# The patch is restricted to faiss/IndexMIRAGE.cpp, taken against the pinned
# revision in native/mirage_bridge/patches/series, and must apply cleanly to a
# fresh checkout of that revision. Commit the resulting .patch file.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_tree="${1:-${MIRAGE_SOURCE:-/root/mirage}}"
patches="$root/native/mirage_bridge/patches"
read -r name revision < <(grep -v '^#' "$patches/series" | grep -v '^[[:space:]]*$' | head -n 1)
target="$patches/$name"

git -C "$source_tree" rev-parse --verify "$revision^{commit}" >/dev/null
changed="$(git -C "$source_tree" diff --name-only "$revision" -- ':(exclude)build')"
if [[ "$changed" != "faiss/IndexMIRAGE.cpp" ]]; then
  echo "Expected only faiss/IndexMIRAGE.cpp to differ from $revision; found:" >&2
  echo "${changed:-<no changes>}" >&2
  exit 1
fi
git -C "$source_tree" diff --binary "$revision" -- faiss/IndexMIRAGE.cpp > "$target.tmp"

# Prove the patch applies to a pristine checkout of the pinned revision.
scratch="$(mktemp -d)"
trap 'rm -rf "$scratch" "$target.tmp"; git -C "$source_tree" worktree prune' EXIT
git -C "$source_tree" worktree add --detach "$scratch/tree" "$revision" >/dev/null
git -C "$scratch/tree" apply --check "$target.tmp"
git -C "$source_tree" worktree remove --force "$scratch/tree"

mv "$target.tmp" "$target"
sha256sum "$target"
echo "Wrote $target; commit it together with native/mirage_bridge/patches/series."
