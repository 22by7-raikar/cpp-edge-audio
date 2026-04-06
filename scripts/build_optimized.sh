#!/usr/bin/env bash
# build_optimized.sh
# Production build with all M6 optimization flags enabled.
#
# Builds a separate build directory (build_opt/) so the standard debug/release
# build in build/ is not overwritten.
#
# Usage:
#   bash scripts/build_optimized.sh           # native + fast-math + LTO
#   bash scripts/build_optimized.sh --no-lto  # skip LTO (faster rebuilds)
#   bash scripts/build_optimized.sh --profile # add -pg for gprof profiling
#
# Output binary: runtime/cpp/build_opt/audio_pipeline_opt

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$REPO_ROOT/runtime/cpp/build_opt"

# --- parse flags ---
ENABLE_LTO=ON
ENABLE_NATIVE=ON
ENABLE_FAST_MATH=ON
BUILD_PROFILING=OFF
OUTPUT_NAME="audio_pipeline_opt"

for arg in "$@"; do
    case "$arg" in
        --no-lto)      ENABLE_LTO=OFF ;;
        --no-native)   ENABLE_NATIVE=OFF ;;
        --no-fast-math) ENABLE_FAST_MATH=OFF ;;
        --profile)     BUILD_PROFILING=ON; OUTPUT_NAME="audio_pipeline_prof" ;;
        *)
            echo "Unknown flag: $arg"
            echo "Usage: $0 [--no-lto] [--no-native] [--no-fast-math] [--profile]"
            exit 1
            ;;
    esac
done

echo "Build dir      : $BUILD_DIR"
echo "native         : $ENABLE_NATIVE"
echo "fast-math      : $ENABLE_FAST_MATH"
echo "LTO            : $ENABLE_LTO"
echo "profiling      : $BUILD_PROFILING"
echo "output binary  : $BUILD_DIR/$OUTPUT_NAME"
echo ""

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake "$REPO_ROOT/runtime/cpp" \
    -DCMAKE_BUILD_TYPE=Release \
    -DWHISPER_ROOT="$REPO_ROOT/vendor/whisper.cpp" \
    -DOPTIMIZE_NATIVE="$ENABLE_NATIVE" \
    -DOPTIMIZE_FAST_MATH="$ENABLE_FAST_MATH" \
    -DOPTIMIZE_LTO="$ENABLE_LTO" \
    -DBUILD_PROFILING="$BUILD_PROFILING" \
    -DCMAKE_RUNTIME_OUTPUT_DIRECTORY="$BUILD_DIR"

make -j"$(nproc)" audio_pipeline

# Rename to keep default build separate
if [ -f "$BUILD_DIR/audio_pipeline" ] && [ "$OUTPUT_NAME" != "audio_pipeline" ]; then
    mv "$BUILD_DIR/audio_pipeline" "$BUILD_DIR/$OUTPUT_NAME"
fi

echo ""
echo "Binary: $BUILD_DIR/$OUTPUT_NAME"
echo ""

# Print a short summary of compile flags actually used
echo "Effective compile flags (from CMakeCache):"
grep -E "CMAKE_CXX_FLAGS|OPTIMIZE_|BUILD_PROFILING" "$BUILD_DIR/CMakeCache.txt" \
    | grep -v "^//" | grep -v "^$" || true
