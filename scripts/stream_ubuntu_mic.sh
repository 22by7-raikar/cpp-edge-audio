#!/usr/bin/env bash
# Capture an ALSA microphone locally and feed the raw-PCM robot hearing interface.

set -euo pipefail

usage() {
    cat >&2 <<'EOF'
Usage: stream_ubuntu_mic.sh --demo EXECUTABLE --model MODEL \
  [--alsa-device DEVICE] [--quality-policy rule|learned|hybrid] \
  [--capture-device LABEL]

The ALSA device defaults to "default" and the quality policy defaults to learned.
JSONL events are stdout; script status and backend diagnostics are stderr.
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

DEMO=""
MODEL=""
ALSA_DEVICE="default"
POLICY="learned"
CAPTURE_DEVICE="alsa_default"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --demo) require_value "$1" "${2:-}"; DEMO="$2"; shift 2 ;;
        --model) require_value "$1" "${2:-}"; MODEL="$2"; shift 2 ;;
        --alsa-device) require_value "$1" "${2:-}"; ALSA_DEVICE="$2"; shift 2 ;;
        --quality-policy) require_value "$1" "${2:-}"; POLICY="$2"; shift 2 ;;
        --capture-device) require_value "$1" "${2:-}"; CAPTURE_DEVICE="$2"; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) echo "ERROR: unknown option: $1" >&2; usage; exit 2 ;;
    esac
done

if [[ -z "$DEMO" || -z "$MODEL" ]]; then
    echo "ERROR: --demo and --model are required" >&2
    usage
    exit 2
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
    echo "ERROR: ffmpeg with ALSA input support is required" >&2
    exit 1
fi

trap 'exit 130' INT TERM
echo "[robot-hearing] streaming ALSA device=$ALSA_DEVICE policy=$POLICY" >&2
ffmpeg -hide_banner -loglevel error \
    -f alsa -i "$ALSA_DEVICE" \
    -ar 16000 -ac 1 -f s16le - | "$DEMO" \
    --source stdin \
    --input-rate 16000 \
    --input-channels 1 \
    --quality-policy "$POLICY" \
    --quality-threshold 0.3 \
    --model "$MODEL" \
    --capture-device "$CAPTURE_DEVICE" \
    --transport alsa_pcm
