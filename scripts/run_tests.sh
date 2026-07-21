#!/usr/bin/env bash
# run_tests.sh
# Build and run the pipeline unit tests.
# Usage: bash scripts/run_tests.sh [Debug|Release]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_TYPE="${1:-Release}"
BUILD_DIR="$REPO_ROOT/runtime/cpp/build_tests"

echo "Build type : $BUILD_TYPE"
echo "Build dir  : $BUILD_DIR"

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake .. \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DWHISPER_ROOT="$REPO_ROOT/vendor/whisper.cpp" \
    -DBUILD_TESTS=ON

make -j"$(nproc)" test_pipeline test_quality_model test_quality_gate_runtime quality_model_parity

echo ""
echo "Running tests..."
ctest --output-on-failure
