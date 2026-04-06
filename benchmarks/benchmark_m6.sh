#!/usr/bin/env bash
# benchmark_m6.sh
# M6 optimization benchmark: compares standard vs optimized builds,
# sweeps thread counts for scaling efficiency, and covers all available
# model quantization variants.
#
# Produces JSON bench files in benchmarks/results/m6/ for analysis with:
#   python tools/python/eval/rtf_plot.py benchmarks/results/m6/*.json
#   python tools/python/eval/compare_bench.py benchmarks/results/m6/*.json
#
# Prerequisites:
#   bash scripts/build.sh            # standard binary
#   bash scripts/build_optimized.sh  # optimized binary
#   At least one ggml-*.bin model in vendor/whisper.cpp/models/
#   At least one *.wav in data/
#
# Usage:
#   bash benchmarks/benchmark_m6.sh [wav_file]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN_STD="$REPO_ROOT/runtime/cpp/build/audio_pipeline"
BIN_OPT="$REPO_ROOT/runtime/cpp/build_opt/audio_pipeline_opt"
MODELS_DIR="$REPO_ROOT/vendor/whisper.cpp/models"
RESULTS_DIR="$REPO_ROOT/benchmarks/results/m6"
mkdir -p "$RESULTS_DIR"

# --- validate binaries ---
HAVE_STD=0; [ -f "$BIN_STD" ] && HAVE_STD=1
HAVE_OPT=0; [ -f "$BIN_OPT" ] && HAVE_OPT=1

if [ $HAVE_STD -eq 0 ] && [ $HAVE_OPT -eq 0 ]; then
    echo "ERROR: neither standard nor optimized binary found."
    echo "  Run: bash scripts/build.sh"
    echo "  Run: bash scripts/build_optimized.sh"
    exit 1
fi

# --- select wav ---
if [ $# -ge 1 ]; then
    WAV_FILES=("$1")
else
    mapfile -t WAV_FILES < <(find "$REPO_ROOT/data" -name "*.wav" | sort)
fi

if [ ${#WAV_FILES[@]} -eq 0 ]; then
    echo "No WAV files in data/. Add audio and re-run."
    exit 0
fi

# Use only the first WAV for a focused optimization comparison.
WAV="${WAV_FILES[0]}"
WAV_BASE="$(basename "${WAV%.wav}")"
echo "Audio: $WAV"

# --- models: use all available ggml-*.bin ---
mapfile -t MODELS < <(find "$MODELS_DIR" -name "ggml-*.bin" 2>/dev/null | sort)
if [ ${#MODELS[@]} -eq 0 ]; then
    echo "No models found in $MODELS_DIR."
    echo "Download: bash vendor/whisper.cpp/models/download-ggml-model.sh base.en"
    exit 0
fi
echo "Models: ${#MODELS[@]} found"

# ---------------------------------------------------------------
# Part 1: Standard vs Optimized binary comparison
#   Fixed: first model, gate on, 4 threads
# ---------------------------------------------------------------
echo ""
echo "=== Part 1: Standard vs Optimized binary comparison ==="
PRIMARY_MODEL="${MODELS[0]}"
MODEL_BASE="$(basename "${PRIMARY_MODEL%.bin}")"

for bi in 0 1; do
    if   [ $bi -eq 0 ] && [ $HAVE_STD -eq 1 ]; then
        BIN="$BIN_STD"; BLABEL="std"
    elif [ $bi -eq 1 ] && [ $HAVE_OPT -eq 1 ]; then
        BIN="$BIN_OPT"; BLABEL="opt"
    else
        continue
    fi

    for CHUNK_MS in 5000 10000; do
        LABEL="${WAV_BASE}__${MODEL_BASE}__${BLABEL}__t4__c${CHUNK_MS}__gate_on"
        JSON_OUT="$RESULTS_DIR/${LABEL}.json"
        TSV_OUT="$RESULTS_DIR/${LABEL}.log"
        echo "  $LABEL"
        "$BIN" \
            --input     "$WAV" \
            --model     "$PRIMARY_MODEL" \
            --chunk-ms  "$CHUNK_MS" \
            --threads   4 \
            --log       "$TSV_OUT" \
            --bench-json "$JSON_OUT" \
            2>/dev/null | grep "event=run_end" || true
    done
done

# ---------------------------------------------------------------
# Part 2: Thread-scaling sweep
#   Optimized binary (or standard if no opt build), first model
# ---------------------------------------------------------------
echo ""
echo "=== Part 2: Thread-scaling sweep ==="
SCALE_BIN="$BIN_OPT"; SCALE_LABEL="opt"
if [ $HAVE_OPT -eq 0 ]; then
    SCALE_BIN="$BIN_STD"; SCALE_LABEL="std"
fi

THREAD_COUNTS=(1 2 4 8)
# Only use physical core count as upper bound
MAX_THREADS="$(nproc)"
for T in "${THREAD_COUNTS[@]}"; do
    if [ "$T" -gt "$MAX_THREADS" ]; then
        echo "  Skipping t$T (only $MAX_THREADS cores available)"
        continue
    fi
    LABEL="${WAV_BASE}__${MODEL_BASE}__${SCALE_LABEL}__t${T}__c5000__gate_on"
    JSON_OUT="$RESULTS_DIR/${LABEL}.json"
    TSV_OUT="$RESULTS_DIR/${LABEL}.log"
    echo "  $LABEL"
    "$SCALE_BIN" \
        --input     "$WAV" \
        --model     "$PRIMARY_MODEL" \
        --chunk-ms  5000 \
        --threads   "$T" \
        --log       "$TSV_OUT" \
        --bench-json "$JSON_OUT" \
        2>/dev/null | grep "event=run_end" || true
done

# ---------------------------------------------------------------
# Part 3: Quantization variant sweep
#   Optimized binary, all available models, fixed config
# ---------------------------------------------------------------
echo ""
echo "=== Part 3: Quantization variant sweep ==="
QUANT_BIN="$SCALE_BIN"

for MODEL in "${MODELS[@]}"; do
    MBASE="$(basename "${MODEL%.bin}")"
    LABEL="${WAV_BASE}__${MBASE}__${SCALE_LABEL}__t4__c5000__gate_on__quant"
    JSON_OUT="$RESULTS_DIR/${LABEL}.json"
    TSV_OUT="$RESULTS_DIR/${LABEL}.log"
    echo "  $LABEL"
    "$QUANT_BIN" \
        --input     "$WAV" \
        --model     "$MODEL" \
        --chunk-ms  5000 \
        --threads   4 \
        --log       "$TSV_OUT" \
        --bench-json "$JSON_OUT" \
        2>/dev/null | grep "event=run_end" || true
done

echo ""
echo "M6 benchmark complete."
echo "Results in: $RESULTS_DIR/"
echo ""
echo "Analyse with:"
echo "  python tools/python/eval/rtf_plot.py $RESULTS_DIR/*.json"
echo "  python tools/python/eval/compare_bench.py $RESULTS_DIR/*.json"
