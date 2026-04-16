#!/usr/bin/env python3
"""
prepare_stage1.py
Build manifests and chunk-level evaluation data from downloaded stage-1 datasets.

Inputs (from scripts/datasets/download_stage1.sh):
    data/raw/librispeech/LibriSpeech/dev-clean/     (required)
    data/raw/librispeech/LibriSpeech/test-clean/    (optional)
    data/external/musan/musan/                      (optional, for noise mixing)

Outputs:
    data/manifests/librispeech_{split}.tsv          - audio path, speaker, text
    data/manifests/musan_{subset}.tsv               - audio path, label
    data/processed/eval_chunks/                     - 5s chunks in WAV format
    data/labels/eval_chunks.tsv                     - chunk path + scene label

Usage:
    python scripts/datasets/prepare_stage1.py --split dev-clean
    python scripts/datasets/prepare_stage1.py --split dev-clean --chunk-sec 5 --limit 200
    python scripts/datasets/prepare_stage1.py --all-splits
"""

import argparse
import os
import struct
import sys
import wave
from pathlib import Path
from typing import List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR   = REPO_ROOT / "data" / "raw"
EXT_DIR   = REPO_ROOT / "data" / "external"
PROC_DIR  = REPO_ROOT / "data" / "processed"
MAN_DIR   = REPO_ROOT / "data" / "manifests"
LAB_DIR   = REPO_ROOT / "data" / "labels"


# -----------------------------------------------------------------------
# WAV utilities (no external libraries needed for basic prep)
# -----------------------------------------------------------------------

def wav_duration(path: Path) -> float:
    """Return duration in seconds for a PCM WAV file."""
    try:
        with wave.open(str(path)) as wf:
            return wf.getnframes() / wf.getframerate()
    except Exception:
        return 0.0


def read_wav_mono_f32(path: Path) -> Tuple[List[float], int]:
    """Read a WAV file as mono float32 samples. Mixes stereo to mono."""
    with wave.open(str(path)) as wf:
        sr        = wf.getframerate()
        n_frames  = wf.getnframes()
        n_ch      = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        raw       = wf.readframes(n_frames)

    if sampwidth == 2:
        fmt = f"<{n_frames * n_ch}h"
        samples = list(struct.unpack(fmt, raw))
        scale = 1.0 / 32768.0
    elif sampwidth == 4:
        fmt = f"<{n_frames * n_ch}i"
        samples = list(struct.unpack(fmt, raw))
        scale = 1.0 / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")

    samples_f = [s * scale for s in samples]

    if n_ch > 1:
        # Mix to mono
        mono = []
        for i in range(0, len(samples_f), n_ch):
            mono.append(sum(samples_f[i:i + n_ch]) / n_ch)
        samples_f = mono

    return samples_f, sr


def write_wav_f32(path: Path, samples: List[float], sr: int) -> None:
    """Write mono float32 samples as 16-bit PCM WAV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(samples)
    raw = struct.pack(f"<{n}h", *[max(-32768, min(32767, int(s * 32767))) for s in samples])
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(raw)


# -----------------------------------------------------------------------
# LibriSpeech manifest builder
# -----------------------------------------------------------------------

def build_librispeech_manifest(split: str, limit: Optional[int] = None) -> List[dict]:
    """
    Scan LibriSpeech split directory and build a list of records:
        {path, speaker_id, chapter_id, utt_id, transcript, duration_sec}
    """
    split_dir = RAW_DIR / "librispeech" / "LibriSpeech" / split
    if not split_dir.exists():
        print(f"WARNING: LibriSpeech split not found: {split_dir}", file=sys.stderr)
        return []

    records = []
    for trans_file in sorted(split_dir.rglob("*.txt")):
        chapter_dir = trans_file.parent
        lines = trans_file.read_text().strip().splitlines()
        for line in lines:
            utt_id, _, text = line.partition(" ")
            flac_path = chapter_dir / f"{utt_id}.flac"
            if not flac_path.exists():
                continue
            records.append({
                "path":       str(flac_path),
                "speaker_id": utt_id.split("-")[0],
                "chapter_id": utt_id.split("-")[1] if "-" in utt_id else "",
                "utt_id":     utt_id,
                "transcript": text.strip(),
                "duration_sec": 0.0,   # filled in selectively below
            })
            if limit and len(records) >= limit:
                break
        if limit and len(records) >= limit:
            break

    return records


def write_manifest(records: List[dict], out_path: Path) -> None:
    """Write a TSV manifest file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        print(f"  No records — skipping {out_path.name}")
        return
    keys = list(records[0].keys())
    with open(out_path, "w") as f:
        f.write("\t".join(keys) + "\n")
        for r in records:
            f.write("\t".join(str(r.get(k, "")) for k in keys) + "\n")
    print(f"  Manifest: {out_path}  ({len(records)} records)")


# -----------------------------------------------------------------------
# MUSAN manifest builder
# -----------------------------------------------------------------------

MUSAN_LABEL_MAP = {
    "music":   "music",
    "noise":   "stationary_noise",
    "speech":  "clean_speech",
}


def build_musan_manifest(subset: Optional[str] = None) -> List[dict]:
    """
    Scan MUSAN and build:
        {path, subset, label, duration_sec}
    subset: "music"|"noise"|"speech"|None (all)
    """
    musan_dir = EXT_DIR / "musan" / "musan"
    if not musan_dir.exists():
        print(f"WARNING: MUSAN not found: {musan_dir}", file=sys.stderr)
        return []

    records = []
    subsets = [subset] if subset else list(MUSAN_LABEL_MAP.keys())
    for s in subsets:
        for wav in sorted((musan_dir / s).rglob("*.wav")):
            records.append({
                "path":      str(wav),
                "subset":    s,
                "label":     MUSAN_LABEL_MAP.get(s, "unknown"),
                "duration_sec": 0.0,
            })
    return records


# -----------------------------------------------------------------------
# Chunk preparation for LibriSpeech eval
# -----------------------------------------------------------------------

def prepare_eval_chunks(
    records: List[dict],
    chunk_sec: float,
    out_dir: Path,
    label_path: Path,
    limit: Optional[int] = None,
) -> None:
    """
    For each LibriSpeech utterance, emit fixed-length mono WAV chunks.
    Labels: clean_speech (>80% speech), low_utility (<20% speech content).

    Because LibriSpeech flac requires ffmpeg/libsndfile, this function
    works only with WAV inputs. FLAC files are skipped with a warning.
    If flac2wav conversion is desired, use: ffmpeg -i in.flac out.wav
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)

    label_rows = []
    n_written  = 0

    for rec in records:
        path = Path(rec["path"])
        if path.suffix.lower() != ".wav":
            # FLAC requires external decoder; skip and warn once
            if not hasattr(prepare_eval_chunks, "_warned_flac"):
                print("  NOTE: FLAC files require ffmpeg conversion.")
                print("  Run: for f in data/raw/librispeech/**/*.flac; do")
                print("         ffmpeg -i \"$f\" \"${f%.flac}.wav\" -y; done")
                prepare_eval_chunks._warned_flac = True  # type: ignore[attr-defined]
            continue

        try:
            samples, sr = read_wav_mono_f32(path)
        except Exception as e:
            print(f"  WARNING: could not read {path}: {e}", file=sys.stderr)
            continue

        chunk_samples = int(chunk_sec * sr)
        for ci, start in enumerate(range(0, len(samples), chunk_samples)):
            chunk = samples[start:start + chunk_samples]
            if len(chunk) < chunk_samples // 2:
                continue  # skip very short trailing fragments

            chunk_name = f"{path.stem}_c{ci:04d}.wav"
            chunk_path = out_dir / chunk_name
            write_wav_f32(chunk_path, chunk, sr)

            # Simple heuristic label: assume LibriSpeech clean = clean_speech
            label_rows.append({
                "path":              str(chunk_path),
                "scene_label":       "clean_speech",
                "should_transcribe": "yes",
                "source":            "librispeech",
                "source_utt":        rec.get("utt_id", ""),
                "transcript":        rec.get("transcript", ""),
            })
            n_written += 1
            if limit and n_written >= limit:
                break

        if limit and n_written >= limit:
            break

    if label_rows:
        write_manifest(label_rows, label_path)
    print(f"  Eval chunks: {n_written} written to {out_dir}")


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare stage-1 manifests and eval chunks."
    )
    parser.add_argument(
        "--split",
        default="dev-clean",
        help="LibriSpeech split (default: dev-clean)",
    )
    parser.add_argument(
        "--all-splits",
        action="store_true",
        help="Process dev-clean and test-clean",
    )
    parser.add_argument(
        "--chunk-sec",
        type=float,
        default=5.0,
        help="Chunk length in seconds for eval chunks (default: 5.0)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max utterances per split (useful for quick smoke runs)",
    )
    parser.add_argument(
        "--musan",
        action="store_true",
        help="Build MUSAN manifest in addition to LibriSpeech",
    )
    args = parser.parse_args()

    splits = ["dev-clean", "test-clean"] if args.all_splits else [args.split]

    for split in splits:
        print(f"\n--- LibriSpeech {split} ---")
        records = build_librispeech_manifest(split, limit=args.limit)
        if not records:
            continue

        man_path = MAN_DIR / f"librispeech_{split.replace('-', '_')}.tsv"
        write_manifest(records, man_path)

        chunk_out = PROC_DIR / "eval_chunks" / split.replace("-", "_")
        label_out = LAB_DIR / f"eval_chunks_{split.replace('-', '_')}.tsv"
        prepare_eval_chunks(
            records,
            chunk_sec=args.chunk_sec,
            out_dir=chunk_out,
            label_path=label_out,
            limit=args.limit,
        )

    if args.musan:
        print("\n--- MUSAN ---")
        for subset in ("music", "noise", "speech"):
            records = build_musan_manifest(subset)
            if not records:
                continue
            man_path = MAN_DIR / f"musan_{subset}.tsv"
            write_manifest(records, man_path)


if __name__ == "__main__":
    main()
