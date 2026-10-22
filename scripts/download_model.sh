#!/usr/bin/env bash
# Download a whisper.cpp GGML model into models/.
#
#   scripts/download_model.sh [tiny.en|base.en|small.en|...]   (default: base.en)

set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$repo_root/models"
bash "$repo_root/vendor/whisper.cpp/models/download-ggml-model.sh" "${1:-base.en}" "$repo_root/models"
