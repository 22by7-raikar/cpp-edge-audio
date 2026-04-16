#!/usr/bin/env bash
# download_stage1.sh
# Download stage-1 datasets for audio ML evaluation.
#
# Datasets:
#   librispeech   - dev-clean and test-clean (default); train-clean-100 if --full
#   musan         - music, noise, speech (background noise source)
#   demand        - diverse environments noise (background noise source)
#   rirs_noises   - room impulse responses for reverberation
#   tau2022       - TAU Urban Acoustic Scenes 2022 (placeholder; requires manual download)
#
# Modes:
#   --lite  : dev-clean + MUSAN only (suitable for laptop, ~3 GB)
#   --full  : all datasets including train-clean-100 (~30 GB total)
#   (default) same as --lite
#
# Usage:
#   bash scripts/datasets/download_stage1.sh [--lite|--full] [--skip-existing]
#
# Outputs go to data/raw/ and data/external/.
# Provenance is recorded in data/manifests/download_provenance.tsv.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RAW_DIR="$REPO_ROOT/data/raw"
EXT_DIR="$REPO_ROOT/data/external"
MANIFEST_DIR="$REPO_ROOT/data/manifests"

mkdir -p "$RAW_DIR" "$EXT_DIR" "$MANIFEST_DIR"

MODE="lite"
SKIP_EXISTING=0

for arg in "$@"; do
    case "$arg" in
        --lite)          MODE="lite"  ;;
        --full)          MODE="full"  ;;
        --skip-existing) SKIP_EXISTING=1 ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

echo "Mode         : $MODE"
echo "Raw dir      : $RAW_DIR"
echo "External dir : $EXT_DIR"
echo "Skip existing: $SKIP_EXISTING"
echo ""

PROV_FILE="$MANIFEST_DIR/download_provenance.tsv"
if [ ! -f "$PROV_FILE" ]; then
    printf "dataset\tsplit\turl\tlocal_path\tstatus\ttimestamp\n" > "$PROV_FILE"
fi

record_provenance() {
    local ds="$1" split="$2" url="$3" local_path="$4" status="$5"
    printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$ds" "$split" "$url" "$local_path" "$status" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        >> "$PROV_FILE"
}

# -------------------------------------------------------
# Utility: download + extract a tar.gz if not already present
# -------------------------------------------------------
fetch_tgz() {
    local ds="$1" split="$2" url="$3" dest_dir="$4" marker="$5"
    local archive_name
    archive_name="$(basename "$url")"
    local archive_path="$dest_dir/$archive_name"

    if [ -d "$dest_dir/$marker" ] && [ "$SKIP_EXISTING" = "1" ]; then
        echo "  [skip] $ds/$split already exists at $dest_dir/$marker"
        record_provenance "$ds" "$split" "$url" "$dest_dir/$marker" "skipped"
        return 0
    fi

    echo "  Downloading $ds/$split ..."
    if ! wget -q --show-progress -P "$dest_dir" "$url"; then
        echo "  ERROR: wget failed for $url"
        record_provenance "$ds" "$split" "$url" "$dest_dir/$marker" "download_failed"
        return 1
    fi

    echo "  Extracting $archive_name ..."
    tar -xzf "$archive_path" -C "$dest_dir"
    rm -f "$archive_path"
    record_provenance "$ds" "$split" "$url" "$dest_dir/$marker" "ok"
    echo "  Done: $dest_dir/$marker"
}

# -------------------------------------------------------
# LibriSpeech
# -------------------------------------------------------
LIBRI_BASE="https://www.openslr.org/resources/12"
LIBRI_DIR="$RAW_DIR/librispeech"
mkdir -p "$LIBRI_DIR"

echo "=== LibriSpeech ==="
fetch_tgz "librispeech" "dev-clean"  "$LIBRI_BASE/dev-clean.tar.gz"   "$LIBRI_DIR" "LibriSpeech/dev-clean"
fetch_tgz "librispeech" "test-clean" "$LIBRI_BASE/test-clean.tar.gz"  "$LIBRI_DIR" "LibriSpeech/test-clean"

if [ "$MODE" = "full" ]; then
    fetch_tgz "librispeech" "train-clean-100" \
        "$LIBRI_BASE/train-clean-100.tar.gz" "$LIBRI_DIR" "LibriSpeech/train-clean-100"
fi

# -------------------------------------------------------
# MUSAN (music, noise, speech)
# -------------------------------------------------------
MUSAN_DIR="$EXT_DIR/musan"
mkdir -p "$MUSAN_DIR"

echo ""
echo "=== MUSAN ==="
fetch_tgz "musan" "all" \
    "https://www.openslr.org/resources/17/musan.tar.gz" \
    "$MUSAN_DIR" "musan"

# -------------------------------------------------------
# DEMAND (diverse environments noise)
# Only in full mode due to size; lite mode records placeholder.
# -------------------------------------------------------
DEMAND_DIR="$EXT_DIR/demand"
mkdir -p "$DEMAND_DIR"

echo ""
echo "=== DEMAND ==="
if [ "$MODE" = "full" ]; then
    # DEMAND is distributed per-environment; download a representative subset
    DEMAND_BASE="https://zenodo.org/record/1227121/files"
    for env in DKITCHEN DLIVING OMEETING OOFFICE SPSQUARE; do
        fetch_tgz "demand" "$env" \
            "$DEMAND_BASE/${env}_16k.zip" \
            "$DEMAND_DIR" "$env" || true
    done
else
    echo "  [lite] DEMAND skipped in lite mode."
    echo "  To download: run with --full or manually place WAVs into $DEMAND_DIR"
    record_provenance "demand" "all" "https://zenodo.org/record/1227121" \
        "$DEMAND_DIR" "skipped_lite"
fi

# -------------------------------------------------------
# RIRS_NOISES (room impulse responses)
# -------------------------------------------------------
RIRS_DIR="$EXT_DIR/rirs_noises"
mkdir -p "$RIRS_DIR"

echo ""
echo "=== RIRS_NOISES ==="
if [ "$MODE" = "full" ]; then
    fetch_tgz "rirs_noises" "all" \
        "https://www.openslr.org/resources/28/rirs_noises.zip" \
        "$RIRS_DIR" "RIRS_NOISES" || true
else
    echo "  [lite] RIRS_NOISES skipped in lite mode."
    record_provenance "rirs_noises" "all" "https://www.openslr.org/resources/28/rirs_noises.zip" \
        "$RIRS_DIR" "skipped_lite"
fi

# -------------------------------------------------------
# TAU Urban Acoustic Scenes 2022 (placeholder)
# Requires DCASE account; cannot be auto-downloaded.
# -------------------------------------------------------
TAU_DIR="$EXT_DIR/tau2022"
mkdir -p "$TAU_DIR"

echo ""
echo "=== TAU Urban Acoustic Scenes 2022 (placeholder) ==="
echo "  TODO: TAU2022 requires manual download."
echo "  1. Register at https://zenodo.org/record/6337421"
echo "  2. Download TAU-urban-acoustic-scenes-2022-mobile-development.audio.*.zip"
echo "  3. Extract into: $TAU_DIR"
record_provenance "tau2022" "development" \
    "https://zenodo.org/record/6337421" \
    "$TAU_DIR" "manual_required"

# -------------------------------------------------------
# Summary
# -------------------------------------------------------
echo ""
echo "Download complete. Provenance: $PROV_FILE"
echo "Next step: bash scripts/datasets/prepare_stage1.py --help"
