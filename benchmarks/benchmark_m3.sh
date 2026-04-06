#!/usr/bin/env bash
# benchmark_m3.sh
# Full benchmark sweep: models x thread-counts x chunk-sizes x gate-on/off.
# Results are written as JSON to benchmarks/results/ for comparison with
# tools/python/eval/compare_bench.py.
#
# Prerequisites:
#   - bash scripts/build.sh
#   - whisper models in vendor/whisper.cpp/models/ (any ggml-*.bin files)
#   - test audio in data/*.wav
#
# Usage:
#   bash benchmarks/benchmark_m3.sh [wav_file]
#
# If wav_file is specified, only that file is processed.
# Otherwise all data/*.wav files are swept.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BINARY="$REPO_ROOT/runtime/cpp/build/audio_pipeline"
MODELS_DIR="$REPO_ROOT/vendor/whisper.cpp/models"
RESULTS_DIR="$REPO_ROOT/benchmarks/results"
mkdir -p "$RESULTS_DIR"

if [ ! -f "$BINARY" ]; then
    echo "ERROR: binary not found. Run: bash scripts/build.sh"
    exit 1
fi

# --- config axes ---
THREAD_COUNTS=(1 2 4)
CHUNK_SIZES=(3000 5000 10000)
GATE_FLAGS=("" "--no-gate")
GATE_LABELS=("gate_on" "gate_off")

# --- select wav files ---
if [ $# -ge 1 ]; then
    WAV_FILES=("$1")
else
    mapfile -t WAV_FILES < <(find "$REPO_ROOT/data" -name "*.wav" | sort)
fi

if [ ${#WAV_FILES[@]} -eq 0 ]; then
    echo "No WAV files found in data/. Add audio files and re-run."
    exit 0
fi

# --- select models ---
mapfile -t MODELS < <(find "$MODELS_DIR" -name "ggml-*.bin" 2>/dev/null | sort)
if [ ${#MODELS[@]} -eq 0 ]; then
    echo "No models found in $MODELS_DIR."
    echo "Download one with: bash vendor/whisper.cpp/models/download-ggml-model.sh base.en"
    exit 0
fi

echo "Models  : ${#MODELS[@]}"
echo "WAV files: ${#WAV_FILES[@]}"
echo "Threads : ${THREAD_COUNTS[*]}"
echo "Chunks  : ${CHUNK_SIZES[*]}ms"
echo "Results : $RESULTS_DIR"
echo ""

TOTAL=0
DONE=0

# Count total runs
for _m in "${MODELS[@]}"; do
    for _w in "${WAV_FILES[@]}"; do
        for _t in "${THREAD_COUNTS[@]}"; do
            for _c in "${CHUNK_SIZES[@]}"; do
                for _g in "${GATE_FLAGS[@]}"; do
                    TOTAL=$((TOTAL + 1))
                done
            done
        done
    done
done
echo "Total runs: $TOTAL"
echo ""

for MODEL in "${MODELS[@]}"; do
    MODEL_BASE="$(basename "${MODEL%.bin}")"

    for WAV in "${WAV_FILES[@]}"; do
        WAV_BASE="$(basename "${WAV%.wav}")"

        for THREADS in "${THREAD_COUNTS[@]}"; do

            for CHUNK_MS in "${CHUNK_SIZES[@]}"; do

                for gi in 0 1; do
                    GATE_FLAG="${GATE_FLAGS[$gi]}"
                    GATE_LABEL="${GATE_LABELS[$gi]}"

                    LABEL="${WAV_BASE}__${MODEL_BASE}__t${THREADS}__c${CHUNK_MS}__${GATE_LABEL}"
                    JSON_OUT="$RESULTS_DIR/${LABEL}.json"
                    TSV_OUT="$RESULTS_DIR/${LABEL}.log"

                    DONE=$((DONE + 1))
                    echo "[${DONE}/${TOTAL}] $LABEL"

                    "$BINARY" \
                        --input    "$WAV" \
                        --model    "$MODEL" \
                        --chunk-ms "$CHUNK_MS" \
                        --threads  "$THREADS" \
                        --log      "$TSV_OUT" \
                        --bench-json "$JSON_OUT" \
                        $GATE_FLAG \
                        2>/dev/null \
                        | grep "event=run_end" || true

                done
            done
        done
    done
done

echo ""
echo "Benchmark complete."
echo "JSON results in: $RESULTS_DIR/"
echo "Compare with:    python tools/python/eval/compare_bench.py $RESULTS_DIR/*.json"
