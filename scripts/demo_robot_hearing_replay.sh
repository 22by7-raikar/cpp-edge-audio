#!/usr/bin/env bash
# Replay a WAV through the raw-PCM robot hearing interface.

set -euo pipefail

usage() {
    cat >&2 <<'EOF'
Usage: demo_robot_hearing_replay.sh --input WAV --demo EXECUTABLE --model MODEL \
  [--quality-policy rule|learned|hybrid] [--capture-device LABEL] [--fast]

Streams mono 16 kHz S16LE PCM to robot_hearing_demo. JSONL events are stdout;
script status and backend diagnostics are stderr. Replay is real-time by default.
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
POLICY="learned"
CAPTURE_DEVICE="wav_replay"
REALTIME=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input) require_value "$1" "${2:-}"; INPUT_WAV="$2"; shift 2 ;;
        --demo) require_value "$1" "${2:-}"; DEMO="$2"; shift 2 ;;
        --model) require_value "$1" "${2:-}"; MODEL="$2"; shift 2 ;;
        --quality-policy) require_value "$1" "${2:-}"; POLICY="$2"; shift 2 ;;
        --capture-device) require_value "$1" "${2:-}"; CAPTURE_DEVICE="$2"; shift 2 ;;
        --fast) REALTIME=0; shift ;;
        --help|-h) usage; exit 0 ;;
        *) echo "ERROR: unknown option: $1" >&2; usage; exit 2 ;;
    esac
done

if [[ -z "$INPUT_WAV" || -z "$DEMO" || -z "$MODEL" ]]; then
    echo "ERROR: --input, --demo, and --model are required" >&2
    usage
    exit 2
fi
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
if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "ERROR: ffmpeg is required" >&2
    exit 1
fi

ffmpeg_args=( -hide_banner -loglevel error )
if [[ "$REALTIME" -eq 1 ]]; then
    ffmpeg_args+=( -re )
fi
ffmpeg_args+=( -i "$INPUT_WAV" -ar 16000 -ac 1 -f s16le - )

echo "[robot-hearing] replay policy=$POLICY realtime=$REALTIME input=$INPUT_WAV" >&2
ffmpeg "${ffmpeg_args[@]}" | "$DEMO" \
    --source stdin \
    --input-rate 16000 \
    --input-channels 1 \
    --quality-policy "$POLICY" \
    --quality-threshold 0.3 \
    --model "$MODEL" \
    --capture-device "$CAPTURE_DEVICE" \
    --transport local_pipe
