#!/usr/bin/env bash
# Run the pipeline on a WAV file.
# Run from repo root: bash scripts/run.sh <wav_file> <model_bin> [extra args]
#
# Example:
#   bash scripts/run.sh data/sample.wav vendor/whisper.cpp/models/ggml-base.en.bin

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BINARY="$REPO_ROOT/runtime/cpp/build/audio_pipeline"
WAV="${1:?Usage: run.sh <wav_file> <model_bin> [args...]}"
MODEL="${2:?Usage: run.sh <wav_file> <model_bin> [args...]}"
shift 2

if [ ! -f "$BINARY" ]; then
    echo "Binary not found. Run: bash scripts/build.sh first."
    exit 1
fi

"$BINARY" --input "$WAV" --model "$MODEL" "$@"
