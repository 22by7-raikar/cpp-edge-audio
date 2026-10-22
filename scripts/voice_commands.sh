#!/usr/bin/env bash
# Feed a WAV file or a live microphone to voice_command_stream.
#
#   scripts/voice_commands.sh --model models/ggml-base.en.bin --wav commands.wav [--realtime]
#   scripts/voice_commands.sh --model models/ggml-base.en.bin --mic [alsa-device]
#   options: --policy rule|learned|hybrid (default: rule)
#
# Audio is converted to mono 16 kHz S16LE with ffmpeg. Command events (JSON
# Lines) go to stdout; diagnostics go to stderr.

set -euo pipefail

usage() {
    sed -n '2,9p' "$0" >&2
    exit 2
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
stream="${VOICE_COMMAND_STREAM:-$repo_root/build/release/apps/voice_command_stream}"
model="" wav="" mic="" policy="rule" realtime=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) model="${2:?}"; shift 2 ;;
        --wav) wav="${2:?}"; shift 2 ;;
        --mic)
            if [[ $# -ge 2 && "$2" != --* ]]; then mic="$2"; shift 2; else mic="default"; shift; fi ;;
        --policy) policy="${2:?}"; shift 2 ;;
        --realtime) realtime=1; shift ;;
        *) usage ;;
    esac
done
[[ -n "$model" && ( -n "$wav" || -n "$mic" ) ]] || usage
[[ -x "$stream" ]] || {
    echo "error: build first: cmake --preset release && cmake --build --preset release" >&2
    exit 1
}
command -v ffmpeg >/dev/null || { echo "error: ffmpeg is required" >&2; exit 1; }

if [[ -n "$wav" ]]; then
    source_args=(-i "$wav")
    if [[ "$realtime" -eq 1 ]]; then source_args=(-re "${source_args[@]}"); fi
    device="file" transport="wav_replay"
else
    source_args=(-f alsa -i "$mic")
    device="alsa:$mic" transport="alsa_pcm"
fi

ffmpeg -hide_banner -loglevel error "${source_args[@]}" -ar 16000 -ac 1 -f s16le - |
    "$stream" --rate 16000 --channels 1 --model "$model" --policy "$policy" \
        --capture-device "$device" --transport "$transport"
