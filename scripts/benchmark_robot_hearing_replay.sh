#!/usr/bin/env bash
# Capture one reproducible robot-hearing WAV replay with JSONL evidence.

set -euo pipefail

usage() {
    cat >&2 <<'EOF'
Usage: benchmark_robot_hearing_replay.sh --input WAV --demo EXECUTABLE \
  --model MODEL --model-label LABEL --quality-policy rule|learned|hybrid \
  --output-dir DIRECTORY [--capture-device LABEL] [--force]

Runs one fast immutable-WAV replay. Results are stored in a deterministic
subdirectory of OUTPUT-DIR as stdout.jsonl, stderr.log, and invocation.txt.
Existing result directories are refused unless --force is supplied.
EOF
}

require_value() {
    if [[ $# -lt 2 || -z "$2" || "$2" == --* ]]; then
        echo "ERROR: $1 requires a value" >&2
        usage
        exit 2
    fi
}

valid_policy() {
    [[ "$1" == rule || "$1" == learned || "$1" == hybrid ]]
}

INPUT_WAV=""
DEMO=""
MODEL=""
MODEL_LABEL=""
POLICY=""
OUTPUT_DIR=""
CAPTURE_DEVICE="benchmark_replay"
FORCE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input) require_value "$1" "${2:-}"; INPUT_WAV="$2"; shift 2 ;;
        --demo) require_value "$1" "${2:-}"; DEMO="$2"; shift 2 ;;
        --model) require_value "$1" "${2:-}"; MODEL="$2"; shift 2 ;;
        --model-label) require_value "$1" "${2:-}"; MODEL_LABEL="$2"; shift 2 ;;
        --quality-policy) require_value "$1" "${2:-}"; POLICY="$2"; shift 2 ;;
        --output-dir) require_value "$1" "${2:-}"; OUTPUT_DIR="$2"; shift 2 ;;
        --capture-device) require_value "$1" "${2:-}"; CAPTURE_DEVICE="$2"; shift 2 ;;
        --force) FORCE=1; shift ;;
        --help|-h) usage; exit 0 ;;
        *) echo "ERROR: unknown option: $1" >&2; usage; exit 2 ;;
    esac
done

for required in INPUT_WAV DEMO MODEL MODEL_LABEL POLICY OUTPUT_DIR; do
    if [[ -z "${!required}" ]]; then
        echo "ERROR: --${required,,} is required" >&2
        usage
        exit 2
    fi
done
if [[ ! -f "$INPUT_WAV" ]]; then
    echo "ERROR: input WAV not found: $INPUT_WAV" >&2
    exit 1
fi
if [[ ! -x "$DEMO" ]]; then
    echo "ERROR: robot_hearing_demo is not executable: $DEMO" >&2
    exit 1
fi
if [[ ! -f "$MODEL" ]]; then
    echo "ERROR: Whisper model not found: $MODEL" >&2
    exit 1
fi
if ! valid_policy "$POLICY"; then
    echo "ERROR: --quality-policy must be rule, learned, or hybrid" >&2
    exit 2
fi
if [[ ! "$MODEL_LABEL" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "ERROR: --model-label may contain only letters, digits, ., _, and -" >&2
    exit 2
fi
if [[ -z "$CAPTURE_DEVICE" ]]; then
    echo "ERROR: --capture-device must be non-empty" >&2
    exit 2
fi
if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: python3 is required for JSONL validation" >&2
    exit 1
fi

INPUT_STEM="$(basename -- "$INPUT_WAV")"
INPUT_STEM="${INPUT_STEM%.*}"
INPUT_STEM="$(printf '%s' "$INPUT_STEM" | LC_ALL=C tr -c 'A-Za-z0-9._-' '_')"
RESULT_DIR="$OUTPUT_DIR/${INPUT_STEM}_${MODEL_LABEL}_${POLICY}"
STDOUT_LOG="$RESULT_DIR/stdout.jsonl"
STDERR_LOG="$RESULT_DIR/stderr.log"
INVOCATION_LOG="$RESULT_DIR/invocation.txt"

if [[ -e "$RESULT_DIR" && "$FORCE" -ne 1 ]]; then
    echo "ERROR: result directory already exists (use --force to overwrite): $RESULT_DIR" >&2
    exit 1
fi
mkdir -p -- "$RESULT_DIR"

REPLAY_SCRIPT="$(cd -- "$(dirname -- "$0")" && pwd)/demo_robot_hearing_replay.sh"
if [[ ! -x "$REPLAY_SCRIPT" ]]; then
    echo "ERROR: replay helper is not executable: $REPLAY_SCRIPT" >&2
    exit 1
fi

{
    printf 'input=%q\n' "$INPUT_WAV"
    printf 'demo=%q\n' "$DEMO"
    printf 'model=%q\n' "$MODEL"
    printf 'model_label=%q\n' "$MODEL_LABEL"
    printf 'quality_policy=%q\n' "$POLICY"
    printf 'capture_device=%q\n' "$CAPTURE_DEVICE"
    printf 'replay_mode=fast\n'
} > "$INVOCATION_LOG"

echo "[robot-hearing] result_dir=$RESULT_DIR" >&2
if "$REPLAY_SCRIPT" \
    --input "$INPUT_WAV" \
    --demo "$DEMO" \
    --model "$MODEL" \
    --quality-policy "$POLICY" \
    --capture-device "$CAPTURE_DEVICE" \
    --fast \
    > "$STDOUT_LOG" \
    2> "$STDERR_LOG"; then
    DEMO_EXIT=0
else
    DEMO_EXIT=$?
fi

if ! SUMMARY="$(python3 - "$STDOUT_LOG" <<'PY'
import json
import sys

path = sys.argv[1]
events = []
with open(path, encoding="utf-8") as handle:
    for line_number, line in enumerate(handle, 1):
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError as error:
            print(
                f"ERROR: invalid JSONL at {path}:{line_number}: {error}",
                file=sys.stderr,
            )
            raise SystemExit(1)

admitted = sum(bool(event.get("admitted")) for event in events)
asr_ran = sum(bool(event.get("asr_ran")) for event in events)
unknown = sum(event.get("intent") == "UNKNOWN" for event in events)
print(f"events={len(events)} admitted={admitted} asr_ran={asr_ran} unknown={unknown}")
PY
)"; then
    VALIDATION_EXIT=1
else
    VALIDATION_EXIT=0
    echo "[robot-hearing] $SUMMARY" >&2
fi

if [[ "$DEMO_EXIT" -ne 0 ]]; then
    echo "ERROR: replay exited with status $DEMO_EXIT" >&2
    exit "$DEMO_EXIT"
fi
if [[ "$VALIDATION_EXIT" -ne 0 ]]; then
    exit "$VALIDATION_EXIT"
fi
