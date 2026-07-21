#!/usr/bin/env bash
# Run the three supported quality policies and keep all output outside tracked results.

set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
    echo "Usage: $0 <input.wav> [whisper-model.bin] [output-directory]" >&2
    exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_WAV="$1"
WHISPER_MODEL="${2:-}"
OUTPUT_DIR="${3:-}"
WHISPER_ROOT="$REPO_ROOT/vendor/whisper.cpp"

if [[ ! -f "$INPUT_WAV" ]]; then
    echo "ERROR: input WAV not found: $INPUT_WAV" >&2
    exit 1
fi
if [[ ! -f "$WHISPER_ROOT/CMakeLists.txt" ]]; then
    echo "ERROR: whisper.cpp submodule is missing: $WHISPER_ROOT" >&2
    exit 1
fi
if [[ -n "$WHISPER_MODEL" && ! -f "$WHISPER_MODEL" ]]; then
    echo "ERROR: Whisper model not found: $WHISPER_MODEL" >&2
    exit 1
fi

if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/quality-gate-demo.XXXXXX")"
else
    if [[ -e "$OUTPUT_DIR" ]]; then
        echo "ERROR: output directory already exists: $OUTPUT_DIR" >&2
        exit 1
    fi
    mkdir -p "$OUTPUT_DIR"
fi

# Keep the Release build under the untracked demo output. This avoids a stale
# executable or an incompatible cache in runtime/cpp/build.
BUILD_DIR="$OUTPUT_DIR/runtime-cpp-release"
PIPELINE_BIN="$BUILD_DIR/audio_pipeline"
cmake -S "$REPO_ROOT/runtime/cpp" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DWHISPER_ROOT="$WHISPER_ROOT"
cmake --build "$BUILD_DIR" --target audio_pipeline -j"$(nproc)"

run_policy() {
    local policy="$1"
    local -a command=(
        "$PIPELINE_BIN"
        --input "$INPUT_WAV"
        --quality-policy "$policy"
        --log "$OUTPUT_DIR/$policy.tsv"
        --bench-json "$OUTPUT_DIR/$policy.json"
    )
    if [[ -n "$WHISPER_MODEL" ]]; then
        command+=(--model "$WHISPER_MODEL")
    else
        command+=(--gate-only)
    fi
    "${command[@]}" \
        >"$OUTPUT_DIR/$policy.stdout" \
        2>"$OUTPUT_DIR/$policy.stderr"
}

run_policy rule
run_policy learned
run_policy hybrid

if [[ -n "$WHISPER_MODEL" ]]; then
    echo "ASR mode: local Whisper model supplied."
else
    echo "ASR mode: gate-only; no Whisper model was supplied."
fi
echo "Demo outputs: $OUTPUT_DIR"
echo "Policies: rule.tsv/json, learned.tsv/json, hybrid.tsv/json"
