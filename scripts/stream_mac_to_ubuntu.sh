#!/usr/bin/env bash
# Capture AVFoundation audio on macOS and stream raw PCM to a remote Ubuntu host.

set -euo pipefail

usage() {
    cat >&2 <<'EOF'
Usage: stream_mac_to_ubuntu.sh --device-index INDEX --ssh-destination USER@HOST \
  --remote-repo PATH --remote-demo PATH --remote-model PATH \
  --quality-policy rule|learned|hybrid --capture-device LABEL
       stream_mac_to_ubuntu.sh --list-devices

REMOTE-DEMO and REMOTE-MODEL may be absolute paths or paths relative to
REMOTE-REPO. JSONL events from the Ubuntu host are stdout; diagnostics are stderr.
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

remote_path() {
    local repo="$1"
    local path="$2"
    if [[ "$path" == /* ]]; then
        printf '%s' "$path"
    else
        printf '%s/%s' "${repo%/}" "$path"
    fi
}

LIST_DEVICES=0
DEVICE_INDEX=""
SSH_DESTINATION=""
REMOTE_REPO=""
REMOTE_DEMO=""
REMOTE_MODEL=""
POLICY=""
CAPTURE_DEVICE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --list-devices) LIST_DEVICES=1; shift ;;
        --device-index) require_value "$1" "${2:-}"; DEVICE_INDEX="$2"; shift 2 ;;
        --ssh-destination) require_value "$1" "${2:-}"; SSH_DESTINATION="$2"; shift 2 ;;
        --remote-repo) require_value "$1" "${2:-}"; REMOTE_REPO="$2"; shift 2 ;;
        --remote-demo) require_value "$1" "${2:-}"; REMOTE_DEMO="$2"; shift 2 ;;
        --remote-model) require_value "$1" "${2:-}"; REMOTE_MODEL="$2"; shift 2 ;;
        --quality-policy) require_value "$1" "${2:-}"; POLICY="$2"; shift 2 ;;
        --capture-device) require_value "$1" "${2:-}"; CAPTURE_DEVICE="$2"; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) echo "ERROR: unknown option: $1" >&2; usage; exit 2 ;;
    esac
done

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "ERROR: ffmpeg is required" >&2
    exit 1
fi
if [[ "$LIST_DEVICES" -eq 1 ]]; then
    if [[ -n "$DEVICE_INDEX$SSH_DESTINATION$REMOTE_REPO$REMOTE_DEMO$REMOTE_MODEL$POLICY$CAPTURE_DEVICE" ]]; then
        echo "ERROR: --list-devices cannot be combined with stream options" >&2
        exit 2
    fi
    # FFmpeg intentionally exits nonzero after AVFoundation device enumeration.
    ffmpeg -hide_banner -f avfoundation -list_devices true -i "" >&2 || true
    exit 0
fi

for required in DEVICE_INDEX SSH_DESTINATION REMOTE_REPO REMOTE_DEMO REMOTE_MODEL POLICY CAPTURE_DEVICE; do
    if [[ -z "${!required}" ]]; then
        echo "ERROR: --${required,,} is required" >&2
        usage
        exit 2
    fi
done
if [[ ! "$DEVICE_INDEX" =~ ^[0-9]+$ ]]; then
    echo "ERROR: --device-index must be a numeric AVFoundation audio index" >&2
    exit 2
fi
if ! valid_policy "$POLICY"; then
    echo "ERROR: --quality-policy must be rule, learned, or hybrid" >&2
    exit 2
fi
if ! command -v ssh >/dev/null 2>&1; then
    echo "ERROR: ssh is required" >&2
    exit 1
fi

DEMO_PATH="$(remote_path "$REMOTE_REPO" "$REMOTE_DEMO")"
MODEL_PATH="$(remote_path "$REMOTE_REPO" "$REMOTE_MODEL")"
printf -v REMOTE_COMMAND 'exec %q --source stdin --input-rate 16000 --input-channels 1 --quality-policy %q --quality-threshold 0.3 --model %q --capture-device %q --transport ssh_pcm' \
    "$DEMO_PATH" "$POLICY" "$MODEL_PATH" "$CAPTURE_DEVICE"

trap 'exit 130' INT TERM
echo "[robot-hearing] streaming AVFoundation index=$DEVICE_INDEX to $SSH_DESTINATION policy=$POLICY" >&2
ffmpeg -hide_banner -loglevel error \
    -f avfoundation -i ":$DEVICE_INDEX" \
    -ar 16000 -ac 1 -f s16le - | \
    ssh -- "$SSH_DESTINATION" "$REMOTE_COMMAND"
