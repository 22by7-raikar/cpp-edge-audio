#!/usr/bin/env bash
# Record deterministic mono 16 kHz PCM WAV audio from macOS AVFoundation.

set -euo pipefail

usage() {
    cat >&2 <<'EOF'
Usage: record_mac_audio.sh --device-index INDEX --output FILE.wav [--force]
       record_mac_audio.sh --list-devices

Captures AVFoundation audio as mono 16 kHz signed 16-bit PCM WAV. No audio
normalization, denoising, or lossy encoding is applied. Press q then Enter to
finish recording cleanly; Ctrl-C also lets FFmpeg finalize the WAV.
EOF
}

require_value() {
    if [[ $# -lt 2 || -z "$2" || "$2" == --* ]]; then
        echo "ERROR: $1 requires a value" >&2
        usage
        exit 2
    fi
}

LIST_DEVICES=0
DEVICE_INDEX=""
OUTPUT=""
FORCE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --list-devices) LIST_DEVICES=1; shift ;;
        --device-index) require_value "$1" "${2:-}"; DEVICE_INDEX="$2"; shift 2 ;;
        --output) require_value "$1" "${2:-}"; OUTPUT="$2"; shift 2 ;;
        --force) FORCE=1; shift ;;
        --help|-h) usage; exit 0 ;;
        *) echo "ERROR: unknown option: $1" >&2; usage; exit 2 ;;
    esac
done

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "ERROR: ffmpeg is required" >&2
    exit 1
fi

if [[ "$LIST_DEVICES" -eq 1 ]]; then
    if [[ -n "$DEVICE_INDEX$OUTPUT" || "$FORCE" -ne 0 ]]; then
        echo "ERROR: --list-devices cannot be combined with capture options" >&2
        exit 2
    fi
    # FFmpeg intentionally exits nonzero after AVFoundation enumeration.
    ffmpeg -hide_banner -f avfoundation -list_devices true -i "" >&2 || true
    exit 0
fi

if [[ -z "$DEVICE_INDEX" || -z "$OUTPUT" ]]; then
    echo "ERROR: --device-index and --output are required" >&2
    usage
    exit 2
fi
if [[ ! "$DEVICE_INDEX" =~ ^[0-9]+$ ]]; then
    echo "ERROR: --device-index must be a numeric AVFoundation audio index" >&2
    exit 2
fi
if [[ "$OUTPUT" != *.wav && "$OUTPUT" != *.WAV ]]; then
    echo "ERROR: --output must end in .wav" >&2
    exit 2
fi
if [[ -e "$OUTPUT" && "$FORCE" -ne 1 ]]; then
    echo "ERROR: output already exists (use --force to overwrite): $OUTPUT" >&2
    exit 1
fi

OUTPUT_PARENT="$(dirname -- "$OUTPUT")"
mkdir -p -- "$OUTPUT_PARENT"

ffmpeg_args=( -hide_banner -loglevel warning )
if [[ "$FORCE" -eq 1 ]]; then
    ffmpeg_args+=( -y )
else
    ffmpeg_args+=( -n )
fi
ffmpeg_args+=(
    -f avfoundation -i ":$DEVICE_INDEX"
    -ar 16000 -ac 1 -c:a pcm_s16le
    "$OUTPUT"
)

echo "[robot-hearing] recording AVFoundation index=$DEVICE_INDEX" >&2
echo "[robot-hearing] press q then Enter to finalize: $OUTPUT" >&2
ffmpeg "${ffmpeg_args[@]}"
