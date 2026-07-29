#!/usr/bin/env bash
# Build the native core and install the extension module next to the Python
# package. Uses plain cargo -- maturin is deliberately not required.
#
# abi3-py311 means one .so works across Python 3.11+, so there is no rebuild
# on interpreter upgrade.
set -euo pipefail
cd "$(dirname "$0")/.."

PROFILE="${1:-release}"
FLAG=""; [ "$PROFILE" = "release" ] && FLAG="--release"

echo ">> cargo test (core)"
cargo test -p tropicam-core --quiet

echo ">> cargo build $PROFILE"
cargo build -p tropicam-py $FLAG --quiet

SO="target/$PROFILE/libtropicam_rs.so"
DEST="src/tropicam_rs.abi3.so"
[ -f "$SO" ] || { echo "expected $SO"; exit 1; }
cp "$SO" "$DEST"
echo ">> installed $DEST ($(du -h "$DEST" | cut -f1))"
