#!/usr/bin/env bash
# Run one WAV through the rule, learned and hybrid admission policies.
#
#   scripts/compare_policies.sh <input.wav> [whisper-model.bin] [output-dir]
#
# Without a model the file is analysed but not transcribed. Writes
# <policy>.log (stdout) and <policy>.json (report) per policy and prints the
# admission summary of each.

set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
    sed -n '2,8p' "$0" >&2
    exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
input="$1"
model="${2:-}"
out_dir="${3:-$(mktemp -d "${TMPDIR:-/tmp}/edge-audio-policies.XXXXXX")}"
pipeline="${AUDIO_PIPELINE:-$repo_root/build/release/apps/audio_pipeline}"

[[ -f "$input" ]] || { echo "error: input not found: $input" >&2; exit 1; }
[[ -x "$pipeline" ]] || {
    echo "error: build first: cmake --preset release && cmake --build --preset release" >&2
    exit 1
}
mkdir -p "$out_dir"

for policy in rule learned hybrid; do
    args=(--input "$input" --policy "$policy" --report "$out_dir/$policy.json")
    if [[ -n "$model" ]]; then args+=(--model "$model"); fi
    "$pipeline" "${args[@]}" >"$out_dir/$policy.log"
    grep '^event=quality_summary' "$out_dir/$policy.log" | tr '\t' '\n' \
        | grep -E '^(policy|learned_probability|final_admission|rejection_reason)=' | paste -sd' '
done
echo "outputs: $out_dir" >&2
