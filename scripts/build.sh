#!/usr/bin/env bash
# Build the C++ pipeline.
# Run from repo root: bash scripts/build.sh [Debug|Release] [cpu|cuda]
#
# Output binary: runtime/cpp/build/audio_pipeline

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_TYPE="${1:-Release}"
MODE="${2:-cpu}"
BUILD_DIR="$REPO_ROOT/runtime/cpp/build"

# whisper.cpp CUDA path needs cmake >= 3.18. Prefer newer local installs.
CMAKE_BIN="${CMAKE_BIN:-}"
if [[ -z "$CMAKE_BIN" ]]; then
    for candidate in /snap/cmake/current/bin/cmake /snap/cmake/1531/bin/cmake; do
        if [[ -x "$candidate" ]]; then
            CMAKE_BIN="$candidate"
            break
        fi
    done
    CMAKE_BIN="${CMAKE_BIN:-cmake}"
fi

if [[ "$MODE" != "cpu" && "$MODE" != "cuda" ]]; then
    echo "ERROR: mode must be cpu or cuda"
    echo "Usage: bash scripts/build.sh [Debug|Release] [cpu|cuda]"
    exit 1
fi

echo "Build type : $BUILD_TYPE"
echo "Build dir  : $BUILD_DIR"
echo "Mode       : $MODE"
echo "cmake      : $CMAKE_BIN ($($CMAKE_BIN --version | head -1))"

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

if [[ "$MODE" == "cuda" ]]; then
    if ! command -v nvcc >/dev/null 2>&1; then
        echo "ERROR: nvcc not found, cannot build cuda mode."
        exit 1
    fi
    CMAKE_CUDA_ARCH="${CMAKE_CUDA_ARCHITECTURES:-native}"
    "$CMAKE_BIN" .. \
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
        -DWHISPER_ROOT="$REPO_ROOT/vendor/whisper.cpp" \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="$CMAKE_CUDA_ARCH"
else
    "$CMAKE_BIN" .. \
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
        -DWHISPER_ROOT="$REPO_ROOT/vendor/whisper.cpp" \
        -DGGML_CUDA=OFF
fi

make -j"$(nproc)"

echo ""
echo "Binary: $BUILD_DIR/audio_pipeline"
