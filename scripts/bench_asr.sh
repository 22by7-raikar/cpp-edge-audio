#!/usr/bin/env bash
# CPU/CUDA benchmark path for tiny/base/small local whisper models.
# Usage:
#   bash scripts/bench_asr.sh [wav_path]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WAV_IN="${1:-}"

RESULTS_DIR="$REPO_ROOT/benchmarks/results/cpu_cuda"
mkdir -p "$RESULTS_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
SUMMARY="$RESULTS_DIR/summary_${TS}.tsv"

if [[ -n "$WAV_IN" ]]; then
    WAV="$WAV_IN"
elif [[ -f "$REPO_ROOT/vendor/whisper.cpp/samples/jfk.wav" ]]; then
    WAV="$REPO_ROOT/vendor/whisper.cpp/samples/jfk.wav"
else
    WAV="$(find "$REPO_ROOT/data/processed/eval_subset" -type f -name "*.wav" | head -1 || true)"
fi

if [[ -z "$WAV" || ! -f "$WAV" ]]; then
    echo "ERROR: no WAV found. Pass one explicitly."
    exit 1
fi

MODELS=()
for m in \
    "$REPO_ROOT/vendor/whisper.cpp/models/ggml-tiny.en.bin" \
    "$REPO_ROOT/vendor/whisper.cpp/models/ggml-base.en.bin" \
    "$REPO_ROOT/vendor/whisper.cpp/models/ggml-small.en.bin"; do
    [[ -f "$m" ]] && MODELS+=("$m")
done

if [[ ${#MODELS[@]} -eq 0 ]]; then
    echo "ERROR: no local tiny/base/small models found."
    echo "Expected one of: ggml-tiny.en.bin, ggml-base.en.bin, ggml-small.en.bin"
    exit 1
fi

HAVE_CUDA=0
if command -v nvcc >/dev/null 2>&1; then
    HAVE_CUDA=1
fi

MODES=(cpu)
if [[ $HAVE_CUDA -eq 1 ]]; then
    MODES+=(cuda)
fi

printf "mode\tmodel\tlatency_ms\trtf\tbackend_requested\tbackend_active\tcpu_fallback\tstdout\tstderr\ttsv_log\tjson_bench\n" > "$SUMMARY"

echo "[bench] wav    : $WAV"
echo "[bench] models : ${#MODELS[@]}"
echo "[bench] modes  : ${MODES[*]}"

audio_pipeline="$REPO_ROOT/runtime/cpp/build/audio_pipeline"
ran_any_mode=0

for mode in "${MODES[@]}"; do
    if ! bash "$REPO_ROOT/scripts/build.sh" Release "$mode"; then
        echo "[bench] WARNING: build failed for mode=$mode, skipping"
        continue
    fi
    ran_any_mode=1

    for model in "${MODELS[@]}"; do
        model_base="$(basename "${model%.bin}")"
        run_id="${mode}__${model_base}__${TS}"
        out_tsv="$RESULTS_DIR/${run_id}.tsv"
        out_json="$RESULTS_DIR/${run_id}.json"
        out_stdout="$RESULTS_DIR/${run_id}.stdout.txt"
        out_stderr="$RESULTS_DIR/${run_id}.stderr.txt"

        "$audio_pipeline" \
            --input "$WAV" \
            --model "$model" \
            --threads 4 \
            --chunk-ms 5000 \
            --log "$out_tsv" \
            --bench-json "$out_json" \
            > "$out_stdout" \
            2> "$out_stderr"

        run_end="$(grep 'event=run_end' "$out_stdout" | tail -1 || true)"
        backend="$(grep -E '\[asr\] backend_requested=' "$out_stderr" | tail -1 || true)"

        latency_ms="$(awk '{for(i=1;i<=NF;i++) if($i ~ /^total_infer_ms=/){split($i,a,"="); print a[2]}}' <<< "$run_end")"
        rtf="$(awk '{for(i=1;i<=NF;i++) if($i ~ /^rtf=/){split($i,a,"="); print a[2]}}' <<< "$run_end")"
        [[ -z "$latency_ms" ]] && latency_ms="NA"
        [[ -z "$rtf" ]] && rtf="NA"

        backend_requested="$(awk '{for(i=1;i<=NF;i++) if($i ~ /^backend_requested=/){split($i,a,"="); print a[2]}}' <<< "$backend")"
        backend_active="$(awk '{for(i=1;i<=NF;i++) if($i ~ /^backend_active=/){split($i,a,"="); print a[2]}}' <<< "$backend")"
        cpu_fallback="$(awk '{for(i=1;i<=NF;i++) if($i ~ /^cpu_fallback=/){split($i,a,"="); print a[2]}}' <<< "$backend")"
        [[ -z "$backend_requested" ]] && backend_requested="unknown"
        [[ -z "$backend_active" ]] && backend_active="unknown"
        [[ -z "$cpu_fallback" ]] && cpu_fallback="unknown"

        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "$mode" "$model_base" "$latency_ms" "$rtf" "$backend_requested" "$backend_active" "$cpu_fallback" \
            "$out_stdout" "$out_stderr" "$out_tsv" "$out_json" >> "$SUMMARY"

        echo "[bench] $mode $model_base latency_ms=$latency_ms rtf=$rtf backend=$backend_active fallback=$cpu_fallback"
    done
done

if [[ $ran_any_mode -eq 0 ]]; then
    echo "ERROR: no mode completed successfully."
    exit 1
fi

echo ""
echo "[bench] summary: $SUMMARY"
