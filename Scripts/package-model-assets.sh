#!/bin/bash
# Packages the JinaV5OmniSmall Core ML bundle into per-component zip archives
# suitable for attaching to a GitHub release as assets.
#
# Each *.mlpackage becomes its own archive (largest is well under the 2 GiB
# per-asset limit); the remaining small entries (manifest.json, tokenizer,
# vision_swift) are grouped into a single "core" archive. Every archive
# contains paths rooted at "JinaV5OmniSmall.bundle/", so unzipping all of
# them into one directory reassembles the complete bundle.
#
# Usage: Scripts/package-model-assets.sh [output-dir]   (default: dist)
#
# Requires the real LFS payloads to be present locally, e.g.:
#   git lfs pull --include "Sources/IndexEngineGloss/Resources/CoreML/JinaV5OmniSmall.bundle/**" --exclude ""
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUNDLE_PARENT="$REPO_ROOT/Sources/IndexEngineGloss/Resources/CoreML"
BUNDLE_NAME="JinaV5OmniSmall.bundle"
OUT_DIR="$(cd "$REPO_ROOT" && mkdir -p "${1:-dist}" && cd "${1:-dist}" && pwd)"

if [[ ! -d "$BUNDLE_PARENT/$BUNDLE_NAME" ]]; then
    echo "error: $BUNDLE_PARENT/$BUNDLE_NAME not found" >&2
    exit 1
fi

# Refuse to package LFS pointer files instead of real payloads.
pointer_hits=$(grep -rl "https://git-lfs.github.com/spec" "$BUNDLE_PARENT/$BUNDLE_NAME" --include="*.bin" 2>/dev/null || true)
first_weight=$(find "$BUNDLE_PARENT/$BUNDLE_NAME" -name "weight.bin" -size -1k | head -1)
if [[ -n "$pointer_hits" || -n "$first_weight" ]]; then
    echo "error: bundle contains LFS pointer files, not real payloads." >&2
    echo "Run: git lfs pull --include \"Sources/IndexEngineGloss/Resources/CoreML/$BUNDLE_NAME/**\" --exclude \"\"" >&2
    exit 1
fi

cd "$BUNDLE_PARENT"

core_entries=()
for entry in "$BUNDLE_NAME"/*; do
    name="$(basename "$entry")"
    if [[ -d "$entry" && "$name" == *.mlpackage ]]; then
        echo "packaging $name"
        rm -f "$OUT_DIR/$BUNDLE_NAME.$name.zip"
        zip -q -r -X "$OUT_DIR/$BUNDLE_NAME.$name.zip" "$entry"
    else
        core_entries+=("$entry")
    fi
done

echo "packaging core (${core_entries[*]})"
rm -f "$OUT_DIR/$BUNDLE_NAME.core.zip"
zip -q -r -X "$OUT_DIR/$BUNDLE_NAME.core.zip" "${core_entries[@]}"

cd "$OUT_DIR"
shasum -a 256 "$BUNDLE_NAME".*.zip > SHA256SUMS
echo
echo "assets written to $OUT_DIR:"
ls -lh "$BUNDLE_NAME".*.zip SHA256SUMS
