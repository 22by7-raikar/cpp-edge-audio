#!/usr/bin/env python3
"""
build_manifests.py
Scan raw datasets and emit JSONL manifests with audio metadata.

Outputs:
    data/manifests/librispeech_dev_clean.jsonl
    data/manifests/librispeech_test_clean.jsonl
    data/manifests/musan_music.jsonl
    data/manifests/musan_noise.jsonl
    data/manifests/musan_speech.jsonl
    data/manifests/rirs.jsonl
    data/manifests/demand_16k.jsonl

Each record (minimum fields):
    path, dataset, split, source_type, sample_rate, duration_sec, channels
LibriSpeech also includes: speaker_id, chapter_id, utterance_id, transcript
MUSAN includes: musan_subset
RIRS includes: rir_type, room_id
DEMAND includes: environment

Usage:
    conda run -n audio_king python scripts/datasets/build_manifests.py
    conda run -n audio_king python scripts/datasets/build_manifests.py --datasets librispeech musan
    conda run -n audio_king python scripts/datasets/build_manifests.py --validate
"""

import argparse
import json
import sys
from pathlib import Path

import soundfile as sf

from dataset_identity import portable_repo_path

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR   = REPO_ROOT / "data" / "raw"
MAN_DIR   = REPO_ROOT / "data" / "manifests"


def _to_rel(path: Path) -> str:
    """Return path relative to REPO_ROOT for portable manifests."""
    return portable_repo_path(path, REPO_ROOT)


# ---------------------------------------------------------------------------
# Audio metadata
# ---------------------------------------------------------------------------

def audio_info(path: Path) -> dict | None:
    """Return {sample_rate, duration_sec, channels} or None on failure."""
    try:
        info = sf.info(str(path))
        return {
            "sample_rate":  info.samplerate,
            "duration_sec": round(info.frames / info.samplerate, 4),
            "channels":     info.channels,
        }
    except Exception as exc:
        return None


# ---------------------------------------------------------------------------
# LibriSpeech
# ---------------------------------------------------------------------------

def build_librispeech(split: str, validate: bool) -> tuple[list[dict], dict]:
    """
    split: "dev-clean" or "test-clean"
    Returns (records, stats).
    LibriSpeech layout: data/raw/librispeech/{split}/LibriSpeech/{split}/
    """
    # Handle double-nested extraction layout
    candidate_dirs = [
        RAW_DIR / "librispeech" / split / "LibriSpeech" / split,
        RAW_DIR / "librispeech" / "LibriSpeech" / split,
        RAW_DIR / "librispeech" / split,
    ]
    split_dir = None
    for c in candidate_dirs:
        if c.exists():
            split_dir = c
            break

    stats = {"found": 0, "skipped": 0, "missing_audio": 0, "unreadable": 0}

    if split_dir is None:
        print(f"  MISSING: LibriSpeech {split} not found under {RAW_DIR / 'librispeech'}",
              file=sys.stderr)
        return [], stats

    records = []
    for trans_file in sorted(split_dir.rglob("*.txt")):
        chapter_dir = trans_file.parent
        for line in trans_file.read_text().strip().splitlines():
            utt_id, _, transcript = line.partition(" ")
            if not utt_id:
                continue
            flac = chapter_dir / f"{utt_id}.flac"
            if not flac.exists():
                stats["missing_audio"] += 1
                continue

            meta = audio_info(flac) if validate else {"sample_rate": None, "duration_sec": None, "channels": None}
            if meta is None:
                print(f"  SKIP (unreadable): {flac}", file=sys.stderr)
                stats["unreadable"] += 1
                stats["skipped"] += 1
                continue

            parts = utt_id.split("-")
            records.append({
                "path":         _to_rel(flac),
                "dataset":      "librispeech",
                "split":        split,
                "source_type":  "clean_speech",
                "sample_rate":  meta["sample_rate"],
                "duration_sec": meta["duration_sec"],
                "channels":     meta["channels"],
                "speaker_id":   parts[0] if len(parts) >= 1 else "",
                "chapter_id":   parts[1] if len(parts) >= 2 else "",
                "utterance_id": utt_id,
                "transcript":   transcript.strip(),
            })
            stats["found"] += 1

    return records, stats


# ---------------------------------------------------------------------------
# MUSAN
# ---------------------------------------------------------------------------

MUSAN_LABELS = {
    "music":  "music",
    "noise":  "stationary_noise",
    "speech": "clean_speech",
}


def build_musan(subset: str, validate: bool) -> tuple[list[dict], dict]:
    """subset: "music" | "noise" | "speech" """
    musan_dir = RAW_DIR / "musan" / subset
    stats = {"found": 0, "skipped": 0, "unreadable": 0}

    if not musan_dir.exists():
        print(f"  MISSING: MUSAN {subset} not found: {musan_dir}", file=sys.stderr)
        return [], stats

    records = []
    for wav in sorted(musan_dir.rglob("*.wav")):
        meta = audio_info(wav) if validate else {"sample_rate": None, "duration_sec": None, "channels": None}
        if meta is None:
            print(f"  SKIP (unreadable): {wav}", file=sys.stderr)
            stats["unreadable"] += 1
            stats["skipped"] += 1
            continue
        records.append({
            "path":         _to_rel(wav),
            "dataset":      "musan",
            "split":        subset,
            "source_type":  MUSAN_LABELS.get(subset, "unknown"),
            "sample_rate":  meta["sample_rate"],
            "duration_sec": meta["duration_sec"],
            "channels":     meta["channels"],
            "musan_subset": subset,
        })
        stats["found"] += 1

    return records, stats


# ---------------------------------------------------------------------------
# RIRS_NOISES
# ---------------------------------------------------------------------------

def build_rirs(validate: bool) -> tuple[list[dict], dict]:
    """
    Scans simulated_rirs (smallroom/mediumroom/largeroom) and
    real_rirs_isotropic_noises. Skips pointsource_noises (not RIRs).
    """
    rirs_root = RAW_DIR / "rirs_noises" / "RIRS_NOISES"
    stats = {"found": 0, "skipped": 0, "unreadable": 0}

    if not rirs_root.exists():
        print(f"  MISSING: RIRS_NOISES not found: {rirs_root}", file=sys.stderr)
        return [], stats

    scan_dirs = {
        "simulated": rirs_root / "simulated_rirs",
        "real":      rirs_root / "real_rirs_isotropic_noises",
    }

    records = []
    for rir_type, d in scan_dirs.items():
        if not d.exists():
            continue
        for wav in sorted(d.rglob("*.wav")):
            meta = audio_info(wav) if validate else {"sample_rate": None, "duration_sec": None, "channels": None}
            if meta is None:
                print(f"  SKIP (unreadable): {wav}", file=sys.stderr)
                stats["unreadable"] += 1
                stats["skipped"] += 1
                continue
            # room_id: parent dir name for simulated, filename stem for real
            room_id = wav.parent.name if rir_type == "simulated" else wav.stem
            records.append({
                "path":         _to_rel(wav),
                "dataset":      "rirs_noises",
                "split":        "all",
                "source_type":  "rir",
                "sample_rate":  meta["sample_rate"],
                "duration_sec": meta["duration_sec"],
                "channels":     meta["channels"],
                "rir_type":     rir_type,
                "room_id":      room_id,
            })
            stats["found"] += 1

    return records, stats


# ---------------------------------------------------------------------------
# DEMAND 16 kHz
# ---------------------------------------------------------------------------

def build_demand(validate: bool) -> tuple[list[dict], dict]:
    """
    DEMAND layout: data/raw/demand_16k/{ENV_16k}/{ENV}/ch??.wav
    e.g. DKITCHEN_16k/DKITCHEN/ch01.wav
    """
    demand_root = RAW_DIR / "demand_16k"
    stats = {"found": 0, "skipped": 0, "unreadable": 0}

    if not demand_root.exists():
        print(f"  MISSING: demand_16k not found: {demand_root}", file=sys.stderr)
        return [], stats

    records = []
    for env_dir in sorted(demand_root.iterdir()):
        if not env_dir.is_dir():
            continue
        env_name = env_dir.name.replace("_16k", "")  # DKITCHEN_16k -> DKITCHEN
        for wav in sorted(env_dir.rglob("*.wav")):
            meta = audio_info(wav) if validate else {"sample_rate": None, "duration_sec": None, "channels": None}
            if meta is None:
                print(f"  SKIP (unreadable): {wav}", file=sys.stderr)
                stats["unreadable"] += 1
                stats["skipped"] += 1
                continue
            records.append({
                "path":         _to_rel(wav),
                "dataset":      "demand_16k",
                "split":        "all",
                "source_type":  "stationary_noise",
                "sample_rate":  meta["sample_rate"],
                "duration_sec": meta["duration_sec"],
                "channels":     meta["channels"],
                "environment":  env_name,
            })
            stats["found"] += 1

    return records, stats


# ---------------------------------------------------------------------------
# Write and report
# ---------------------------------------------------------------------------

def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def print_stats(name: str, records: list[dict], stats: dict) -> None:
    found    = stats["found"]
    skipped  = stats["skipped"]
    unread   = stats["unreadable"]

    if not records:
        print(f"  {name:35s}  0 records  (MISSING or empty)")
        return

    # Sample-rate distribution
    srs = [r["sample_rate"] for r in records if r["sample_rate"] is not None]
    sr_set = set(srs)
    sr_note = f"sr={sorted(sr_set)}" if srs else "sr=unknown"
    if len(sr_set) > 1:
        sr_note += "  *** MISMATCH ***"

    dur_vals = [r["duration_sec"] for r in records if r["duration_sec"] is not None]
    total_h  = sum(dur_vals) / 3600 if dur_vals else 0.0

    print(f"  {name:35s}  {found:5d} files  skip={skipped}  unread={unread}"
          f"  {sr_note}  total={total_h:.1f}h")


def sanity_check_dirs() -> None:
    expected = {
        "librispeech":        RAW_DIR / "librispeech",
        "musan":              RAW_DIR / "musan",
        "rirs_noises":        RAW_DIR / "rirs_noises",
        "demand_16k":         RAW_DIR / "demand_16k",
    }
    print("\nDirectory check:")
    for name, path in expected.items():
        status = "OK" if path.exists() else "MISSING"
        print(f"  {name:20s}  {status}  ({path})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build JSONL manifests for all datasets.")
    parser.add_argument(
        "--datasets", nargs="+",
        choices=["librispeech", "musan", "rirs", "demand", "all"],
        default=["all"],
        help="Which datasets to process (default: all)",
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Read audio headers to populate sample_rate/duration_sec/channels "
             "(slower but thorough). Without this flag, metadata fields are null.",
    )
    args = parser.parse_args()

    do_all        = "all" in args.datasets
    do_librispeech = do_all or "librispeech" in args.datasets
    do_musan      = do_all or "musan" in args.datasets
    do_rirs       = do_all or "rirs" in args.datasets
    do_demand     = do_all or "demand" in args.datasets

    sanity_check_dirs()
    print(f"\nValidate headers: {'yes (reading audio info)' if args.validate else 'no (metadata will be null)'}")
    print()

    all_stats: list[tuple[str, list[dict], dict]] = []

    if do_librispeech:
        for split in ("dev-clean", "test-clean", "train-clean-100"):
            key = split.replace("-", "_")
            print(f"Scanning LibriSpeech {split} ...")
            recs, stats = build_librispeech(split, args.validate)
            if recs:
                out = MAN_DIR / f"librispeech_{key}.jsonl"
                write_jsonl(recs, out)
                print(f"  -> {out}")
            all_stats.append((f"librispeech_{key}", recs, stats))

    if do_musan:
        for subset in ("music", "noise", "speech"):
            print(f"Scanning MUSAN {subset} ...")
            recs, stats = build_musan(subset, args.validate)
            if recs:
                out = MAN_DIR / f"musan_{subset}.jsonl"
                write_jsonl(recs, out)
                print(f"  -> {out}")
            all_stats.append((f"musan_{subset}", recs, stats))

    if do_rirs:
        print("Scanning RIRS_NOISES ...")
        recs, stats = build_rirs(args.validate)
        if recs:
            out = MAN_DIR / "rirs.jsonl"
            write_jsonl(recs, out)
            print(f"  -> {out}")
        all_stats.append(("rirs", recs, stats))

    if do_demand:
        print("Scanning DEMAND 16 kHz ...")
        recs, stats = build_demand(args.validate)
        if recs:
            out = MAN_DIR / "demand_16k.jsonl"
            write_jsonl(recs, out)
            print(f"  -> {out}")
        all_stats.append(("demand_16k", recs, stats))

    print("\n--- Summary ---")
    for name, recs, stats in all_stats:
        print_stats(name, recs, stats)

    total_files = sum(s["found"] for _, _, s in all_stats)
    total_skip  = sum(s["skipped"] for _, _, s in all_stats)
    print(f"\nTotal: {total_files} files indexed, {total_skip} skipped")


if __name__ == "__main__":
    main()
