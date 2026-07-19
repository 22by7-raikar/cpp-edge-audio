#!/usr/bin/env bash
# CUDA/CPU ASR smoke test for whisper.cpp runtime integration.
# Usage:
#   bash scripts/smoke_asr.sh [cpu|cuda] [wav_path] [model_path]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-cpu}"
WAV_IN="${2:-}"
MODEL_IN="${3:-}"

if [[ "$MODE" != "cpu" && "$MODE" != "cuda" ]]; then
    echo "ERROR: mode must be cpu or cuda"
    echo "Usage: bash scripts/smoke_asr.sh [cpu|cuda] [wav_path] [model_path]"
    exit 1
fi

RESULTS_DIR="$REPO_ROOT/benchmarks/results/smoke"
mkdir -p "$RESULTS_DIR"
TS="$(date +%Y%m%d_%H%M%S)"

# Pick a known local WAV when not provided.
if [[ -n "$WAV_IN" ]]; then
    WAV="$WAV_IN"
elif [[ -f "$REPO_ROOT/vendor/whisper.cpp/samples/jfk.wav" ]]; then
    WAV="$REPO_ROOT/vendor/whisper.cpp/samples/jfk.wav"
else
    WAV="$(find "$REPO_ROOT/data/processed/eval_subset" -type f -name "*.wav" | head -1 || true)"
fi

if [[ -z "$WAV" || ! -f "$WAV" ]]; then
    echo "ERROR: no WAV found. Provide one: bash scripts/smoke_asr.sh $MODE <wav> [model]"
    exit 1
fi

# Use tiny/base/small if available; do not auto-download.
if [[ -n "$MODEL_IN" ]]; then
    MODEL="$MODEL_IN"
else
    for m in \
        "$REPO_ROOT/vendor/whisper.cpp/models/ggml-tiny.en.bin" \
        "$REPO_ROOT/vendor/whisper.cpp/models/ggml-base.en.bin" \
        "$REPO_ROOT/vendor/whisper.cpp/models/ggml-small.en.bin"; do
        if [[ -f "$m" ]]; then
            MODEL="$m"
            break
        fi
    done
fi

if [[ -z "${MODEL:-}" || ! -f "$MODEL" ]]; then
    echo "ERROR: no local model found (tiny/base/small)."
    echo "Provide one: bash scripts/smoke_asr.sh $MODE "$WAV" <model_path>"
    exit 1
fi

LOG_OUT="$RESULTS_DIR/smoke_${MODE}_${TS}.tsv"
JSON_OUT="$RESULTS_DIR/smoke_${MODE}_${TS}.json"
STDOUT_OUT="$RESULTS_DIR/smoke_${MODE}_${TS}.stdout.txt"
STDERR_OUT="$RESULTS_DIR/smoke_${MODE}_${TS}.stderr.txt"

echo "[smoke] mode      : $MODE"
echo "[smoke] wav       : $WAV"
echo "[smoke] model     : $MODEL"
echo "[smoke] tsv log   : $LOG_OUT"
echo "[smoke] json bench: $JSON_OUT"

bash "$REPO_ROOT/scripts/build.sh" Release "$MODE"

"$REPO_ROOT/runtime/cpp/build/audio_pipeline" \
    --input "$WAV" \
    --model "$MODEL" \
    --threads 4 \
    --chunk-ms 5000 \
    --log "$LOG_OUT" \
    --bench-json "$JSON_OUT" \
    > "$STDOUT_OUT" \
    2> "$STDERR_OUT"

echo ""
echo "[smoke] backend line:"
grep -E "\[asr\] backend_requested=" "$STDERR_OUT" || echo "backend line not found"

echo "[smoke] run_end line:"
grep "event=run_end" "$STDOUT_OUT" || echo "run_end line not found"

echo "[smoke] stdout: $STDOUT_OUT"
echo "[smoke] stderr: $STDERR_OUT"
