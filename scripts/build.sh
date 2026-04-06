#!/usr/bin/env bash
# Build the C++ pipeline.
# Run from repo root: bash scripts/build.sh [Debug|Release]
#
# Output binary: runtime/cpp/build/audio_pipeline

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_TYPE="${1:-Release}"
BUILD_DIR="$REPO_ROOT/runtime/cpp/build"

echo "Build type : $BUILD_TYPE"
echo "Build dir  : $BUILD_DIR"

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake .. \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DWHISPER_ROOT="$REPO_ROOT/vendor/whisper.cpp"

make -j"$(nproc)"

echo ""
echo "Binary: $BUILD_DIR/audio_pipeline"
