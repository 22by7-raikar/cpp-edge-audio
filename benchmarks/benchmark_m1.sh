# benchmark_m1.sh
# Reference invocation for Milestone 1 benchmarking.
# Logs go to benchmarks/results/ (gitignored).
# Compare: gate-enabled vs gate-disabled, vary chunk size.
#
# Prerequisites:
#   - binary built: bash scripts/build.sh
#   - model downloaded: bash vendor/whisper.cpp/models/download-ggml-model.sh base.en
#   - test audio in data/
#
# Usage: bash benchmarks/benchmark_m1.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BINARY="$REPO_ROOT/runtime/cpp/build/audio_pipeline"
MODEL="$REPO_ROOT/vendor/whisper.cpp/models/ggml-base.en.bin"
RESULTS_DIR="$REPO_ROOT/benchmarks/results"
mkdir -p "$RESULTS_DIR"

run_bench() {
    local label="$1"
    local wav="$2"
    local chunk_ms="$3"
    local threads="$4"
    local extra="${5:-}"
    local logfile="$RESULTS_DIR/${label}.log"

    echo "--- $label ---"
    "$BINARY" \
        --input   "$wav" \
        --model   "$MODEL" \
        --chunk-ms "$chunk_ms" \
        --threads  "$threads" \
        --log      "$logfile" \
        $extra
    echo "Log: $logfile"
}

# Iterate over test files in data/
for wav in "$REPO_ROOT"/data/*.wav; do
    base="$(basename "${wav%.wav}")"

    # Gate enabled, 5s chunks
    run_bench "${base}_gate_on_5s_t4"  "$wav" 5000 4 ""
    # Gate disabled, 5s chunks
    run_bench "${base}_gate_off_5s_t4" "$wav" 5000 4 "--no-gate"
    # Gate enabled, 10s chunks
    run_bench "${base}_gate_on_10s_t4" "$wav" 10000 4 ""
done

echo ""
echo "All benchmark runs complete. Results in $RESULTS_DIR/"
echo "Use tools/python to analyze logs."
