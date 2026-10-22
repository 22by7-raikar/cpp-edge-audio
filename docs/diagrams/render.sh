#!/usr/bin/env bash
# Render every *.mmd in this directory to SVG with the Mermaid CLI.
#   npm install -g @mermaid-js/mermaid-cli   (provides mmdc)
#   docs/diagrams/render.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
puppeteer_args=()
if [[ -n "${PUPPETEER_CONFIG:-}" ]]; then puppeteer_args=(-p "$PUPPETEER_CONFIG"); fi
for source in *.mmd; do
    mmdc "${puppeteer_args[@]}" -i "$source" -o "${source%.mmd}.svg" -b white -t neutral -q
    echo "rendered ${source%.mmd}.svg"
done
