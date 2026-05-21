#!/usr/bin/env python3
"""
build_quality_train.py
Build a larger labeled training set for the learned quality predictor.

Speech-derived labels use librispeech_test_clean, which is completely
disjoint from eval_subset (built from dev-clean).  Music and noise sources
already present in eval_subset are excluded to prevent leakage.

Output:
    data/processed/quality_train/     rendered WAVs (clean_speech: symlinks)
    data/labels/quality_train.jsonl   one record per example

Default target counts (~1700 total):
    clean_speech          400
    speech_in_noise       400
    speech_in_reverb      200
    music                 200
    stationary_noise      150
    clipped_or_distorted  200
    low_utility           150

Usage:
    python scripts/datasets/build_quality_train.py [--overwrite] [--dry-run]
    python scripts/datasets/build_quality_train.py --counts music=100 --dry-run
"""

import argparse
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np

# Shared utilities from build_eval_subset live in the same directory.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from build_eval_subset import (    # noqa: E402
    CHUNK_SEC,
    SNR_MIN,
    SNR_MAX,
    TARGET_SR,
    apply_clipping,
    apply_rir,
    load_manifest,
    load_mono_16k,
    make_label,
    mix_at_snr,
    save_wav,
    trim_or_pad,
)

REPO_ROOT  = Path(__file__).resolve().parents[2]
MAN_DIR    = REPO_ROOT / "data" / "manifests"
OUT_DIR    = REPO_ROOT / "data" / "processed" / "quality_train"
LABEL_PATH = REPO_ROOT / "data" / "labels"   / "quality_train.jsonl"
EVAL_LABEL = REPO_ROOT / "data" / "labels"   / "eval_subset.jsonl"

DEFAULT_COUNTS: dict[str, int] = {
    "clean_speech":         400,
    "speech_in_noise":      400,
    "speech_in_reverb":     200,
    "music":                200,
    "stationary_noise":     150,
    "clipped_or_distorted": 200,
    "low_utility":          150,
}


def load_excluded_sources(path: Path) -> set:
    """Source paths already used in eval_subset — exclude from training pools."""
    if not path.exists():
        return set()
    excluded = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                src = rec.get("source", "")
                if src:
                    excluded.add(src)
    return excluded


def _sample(pool: list, rng: random.Random, n: int) -> list:
    avail = [r for r in pool if Path(r["path"]).exists()]
    if len(avail) < n:
        print(f"  WARN: only {len(avail)} readable records, requested {n}", file=sys.stderr)
    return rng.sample(avail, min(n, len(avail)))


def build_clean_speech(pool: list, rng: random.Random, n: int, seed: int, dry_run: bool) -> list:
    selected = _sample(pool, rng, n)
    labels = []
    for i, rec in enumerate(selected):
        src = Path(rec["path"])
        out_path = OUT_DIR / "clean_speech" / f"clean_speech_{i:04d}{src.suffix}"
        if not dry_run:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if out_path.exists() or out_path.is_symlink():
                out_path.unlink()
            out_path.symlink_to(src)
        labels.append(make_label(
            out_path=out_path, label="clean_speech", should_transcribe="yes",
            synthetic=False, source=rec["path"], source_type="clean_speech",
            base_utterance_id=rec.get("utterance_id", ""),
            corruption_source="", snr_db=None, rir_id="", seed=seed,
            duration_sec=float(rec.get("duration_sec", CHUNK_SEC)),
            sample_rate=int(rec.get("sample_rate", TARGET_SR)),
        ))
    return labels


def build_speech_in_noise(
    pool: list, noise_pool: list, rng: random.Random, n: int, seed: int, dry_run: bool
) -> list:
    selected = _sample(pool, rng, n)
    noise_ok = [r for r in noise_pool if Path(r["path"]).exists()]
    if not noise_ok:
        print("  WARN: no readable noise for speech_in_noise", file=sys.stderr)
        return []
    labels = []
    chunk_samples = int(CHUNK_SEC * TARGET_SR)
    for i, rec in enumerate(selected):
        snr_db = rng.uniform(SNR_MIN, SNR_MAX)
        noise_rec = rng.choice(noise_ok)
        out_path = OUT_DIR / "speech_in_noise" / f"speech_in_noise_{i:04d}.wav"
        if not dry_run:
            sp = load_mono_16k(Path(rec["path"]))
            ns = load_mono_16k(Path(noise_rec["path"]))
            if sp is None or ns is None:
                continue
            save_wav(out_path, mix_at_snr(
                trim_or_pad(sp, chunk_samples),
                trim_or_pad(ns, chunk_samples),
                snr_db,
            ))
        labels.append(make_label(
            out_path=out_path, label="speech_in_noise", should_transcribe="yes",
            synthetic=True, source=rec["path"], source_type="clean_speech",
            base_utterance_id=rec.get("utterance_id", ""),
            corruption_source=noise_rec["path"],
            snr_db=round(snr_db, 1), rir_id="", seed=seed,
            duration_sec=CHUNK_SEC, sample_rate=TARGET_SR,
        ))
    return labels


def build_speech_in_reverb(
    pool: list, rir_pool: list, rng: random.Random, n: int, seed: int, dry_run: bool
) -> list:
    selected = _sample(pool, rng, n)
    rir_ok = [r for r in rir_pool if Path(r["path"]).exists()]
    if not rir_ok:
        print("  WARN: no readable RIRs for speech_in_reverb", file=sys.stderr)
        return []
    labels = []
    chunk_samples = int(CHUNK_SEC * TARGET_SR)
    for i, rec in enumerate(selected):
        rir_rec = rng.choice(rir_ok)
        out_path = OUT_DIR / "speech_in_reverb" / f"speech_in_reverb_{i:04d}.wav"
        if not dry_run:
            sp  = load_mono_16k(Path(rec["path"]))
            rir = load_mono_16k(Path(rir_rec["path"]))
            if sp is None or rir is None:
                continue
            save_wav(out_path, apply_rir(trim_or_pad(sp, chunk_samples), rir))
        labels.append(make_label(
            out_path=out_path, label="speech_in_reverb", should_transcribe="yes",
            synthetic=True, source=rec["path"], source_type="clean_speech",
            base_utterance_id=rec.get("utterance_id", ""),
            corruption_source=rir_rec["path"],
            snr_db=None, rir_id=Path(rir_rec["path"]).stem, seed=seed,
            duration_sec=CHUNK_SEC, sample_rate=TARGET_SR,
        ))
    return labels


def build_music(
    pool: list, rng: random.Random, n: int, seed: int, dry_run: bool, excluded: set
) -> list:
    avail = [r for r in pool if Path(r["path"]).exists() and r["path"] not in excluded]
    if len(avail) < n:
        print(f"  WARN: only {len(avail)} music records after exclusion, requested {n}", file=sys.stderr)
    selected = rng.sample(avail, min(n, len(avail)))
    labels = []
    chunk_samples = int(CHUNK_SEC * TARGET_SR)
    for i, rec in enumerate(selected):
        out_path = OUT_DIR / "music" / f"music_{i:04d}.wav"
        if not dry_run:
            audio = load_mono_16k(Path(rec["path"]))
            if audio is None:
                continue
            save_wav(out_path, trim_or_pad(audio, chunk_samples))
        labels.append(make_label(
            out_path=out_path, label="music", should_transcribe="no",
            synthetic=False, source=rec["path"], source_type="music",
            base_utterance_id="", corruption_source="",
            snr_db=None, rir_id="", seed=seed,
            duration_sec=CHUNK_SEC, sample_rate=TARGET_SR,
        ))
    return labels


def build_stationary_noise(
    noise_pool: list, demand_pool: list,
    rng: random.Random, n: int, seed: int, dry_run: bool, excluded: set,
) -> list:
    pool = (
        [r for r in demand_pool if Path(r["path"]).exists() and r["path"] not in excluded] +
        [r for r in noise_pool  if Path(r["path"]).exists() and r["path"] not in excluded]
    )
    if len(pool) < n:
        print(f"  WARN: only {len(pool)} noise records after exclusion, requested {n}", file=sys.stderr)
    selected = rng.sample(pool, min(n, len(pool)))
    labels = []
    chunk_samples = int(CHUNK_SEC * TARGET_SR)
    for i, rec in enumerate(selected):
        out_path = OUT_DIR / "stationary_noise" / f"stationary_noise_{i:04d}.wav"
        if not dry_run:
            audio = load_mono_16k(Path(rec["path"]))
            if audio is None:
                continue
            save_wav(out_path, trim_or_pad(audio, chunk_samples))
        labels.append(make_label(
            out_path=out_path, label="stationary_noise", should_transcribe="no",
            synthetic=False, source=rec["path"], source_type="stationary_noise",
            base_utterance_id="", corruption_source="",
            snr_db=None, rir_id="", seed=seed,
            duration_sec=CHUNK_SEC, sample_rate=TARGET_SR,
        ))
    return labels


def build_clipped(pool: list, rng: random.Random, n: int, seed: int, dry_run: bool) -> list:
    selected = _sample(pool, rng, n)
    labels = []
    chunk_samples = int(CHUNK_SEC * TARGET_SR)
    for i, rec in enumerate(selected):
        clip_thresh = rng.uniform(0.1, 0.4)
        out_path = OUT_DIR / "clipped_or_distorted" / f"clipped_{i:04d}.wav"
        if not dry_run:
            audio = load_mono_16k(Path(rec["path"]))
            if audio is None:
                continue
            save_wav(out_path, apply_clipping(trim_or_pad(audio, chunk_samples), clip_thresh))
        labels.append(make_label(
            out_path=out_path, label="clipped_or_distorted", should_transcribe="no",
            synthetic=True, source=rec["path"], source_type="clean_speech",
            base_utterance_id=rec.get("utterance_id", ""),
            corruption_source=f"hard_clip@{clip_thresh:.2f}",
            snr_db=None, rir_id="", seed=seed,
            duration_sec=CHUNK_SEC, sample_rate=TARGET_SR,
        ))
    return labels


def build_low_utility(pool: list, rng: random.Random, n: int, seed: int, dry_run: bool) -> list:
    selected = _sample(pool, rng, n)
    labels = []
    chunk_samples = int(CHUNK_SEC * TARGET_SR)
    for i, rec in enumerate(selected):
        out_path = OUT_DIR / "low_utility" / f"low_utility_{i:04d}.wav"
        if not dry_run:
            audio = load_mono_16k(Path(rec["path"]))
            if audio is None:
                continue
            keep = min(int(0.5 * TARGET_SR), len(audio))
            snippet = audio[-keep:] * 0.05
            padded = np.zeros(chunk_samples, dtype=np.float32)
            padded[:keep] = snippet
            save_wav(out_path, padded)
        labels.append(make_label(
            out_path=out_path, label="low_utility", should_transcribe="no",
            synthetic=True, source=rec["path"], source_type="clean_speech",
            base_utterance_id=rec.get("utterance_id", ""),
            corruption_source="silence_pad+attenuation",
            snr_db=None, rir_id="", seed=seed,
            duration_sec=CHUNK_SEC, sample_rate=TARGET_SR,
        ))
    return labels


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--seed",      type=int,  default=123,
                    help="RNG seed (default: 123; eval_subset uses 42)")
    ap.add_argument("--counts",    nargs="+", default=[], metavar="CLASS=N",
                    help="Override per-class counts, e.g. music=100")
    ap.add_argument("--overwrite", action="store_true",
                    help="Delete existing output before regenerating")
    ap.add_argument("--dry-run",   action="store_true",
                    help="Plan only; no writes")
    args = ap.parse_args()

    counts: dict[str, int] = dict(DEFAULT_COUNTS)
    for item in args.counts:
        k, _, v = item.partition("=")
        if k in counts and v:
            counts[k] = int(v)
        else:
            print(f"  WARN: unknown or invalid --counts token '{item}'", file=sys.stderr)

    print(f"Seed:      {args.seed}")
    print(f"Dry run:   {args.dry_run}")
    print(f"Overwrite: {args.overwrite}")
    print(f"Counts:    {counts}")
    print(f"Total target: {sum(counts.values())}")
    print()

    if not args.dry_run:
        if OUT_DIR.exists() and any(OUT_DIR.rglob("*")):
            if not args.overwrite:
                print(
                    f"ERROR: {OUT_DIR} already has data. Use --overwrite.",
                    file=sys.stderr,
                )
                sys.exit(1)
            shutil.rmtree(OUT_DIR)
        if LABEL_PATH.exists() and args.overwrite:
            LABEL_PATH.unlink()

    excluded = load_excluded_sources(EVAL_LABEL)
    print(f"Excluded sources from eval_subset: {len(excluded)}")
    print()

    rng = random.Random(args.seed)

    test_clean  = load_manifest(MAN_DIR / "librispeech_test_clean.jsonl")
    musan_noise = load_manifest(MAN_DIR / "musan_noise.jsonl")
    musan_music = load_manifest(MAN_DIR / "musan_music.jsonl")
    rirs        = load_manifest(MAN_DIR / "rirs.jsonl")
    demand      = load_manifest(MAN_DIR / "demand_16k.jsonl")

    missing = []
    if not test_clean:  missing.append("librispeech_test_clean.jsonl")
    if not musan_noise: missing.append("musan_noise.jsonl")
    if not musan_music: missing.append("musan_music.jsonl")
    if not rirs:        missing.append("rirs.jsonl")
    if not demand:      missing.append("demand_16k.jsonl")
    if missing:
        print(f"ERROR: missing manifests: {missing}", file=sys.stderr)
        sys.exit(1)

    sim_rirs = [r for r in rirs if r.get("rir_type") == "simulated"]

    print(f"test-clean:      {len(test_clean):5d} records")
    print(f"musan/noise:     {len(musan_noise):5d} records")
    print(f"musan/music:     {len(musan_music):5d} records")
    print(f"simulated RIRs:  {len(sim_rirs):5d} records")
    print(f"demand_16k:      {len(demand):5d} records")
    print()

    seed = args.seed
    all_labels: list = []

    print("Building clean_speech ...")
    all_labels += build_clean_speech(test_clean, rng, counts["clean_speech"], seed, args.dry_run)

    print("Building speech_in_noise ...")
    all_labels += build_speech_in_noise(
        test_clean, musan_noise + demand, rng, counts["speech_in_noise"], seed, args.dry_run)

    print("Building speech_in_reverb ...")
    all_labels += build_speech_in_reverb(
        test_clean, sim_rirs, rng, counts["speech_in_reverb"], seed, args.dry_run)

    print("Building music ...")
    all_labels += build_music(musan_music, rng, counts["music"], seed, args.dry_run, excluded)

    print("Building stationary_noise ...")
    all_labels += build_stationary_noise(
        musan_noise, demand, rng, counts["stationary_noise"], seed, args.dry_run, excluded)

    print("Building clipped_or_distorted ...")
    all_labels += build_clipped(test_clean, rng, counts["clipped_or_distorted"], seed, args.dry_run)

    print("Building low_utility ...")
    all_labels += build_low_utility(test_clean, rng, counts["low_utility"], seed, args.dry_run)

    if not args.dry_run:
        LABEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LABEL_PATH, "w") as f:
            for rec in all_labels:
                f.write(json.dumps(rec) + "\n")
        print(f"\nLabels written: {LABEL_PATH}  ({len(all_labels)} records)")
        print(f"Audio written:  {OUT_DIR}/")
    else:
        print(f"\nDry run complete. Would generate {len(all_labels)} examples.")

    print("\n--- Class counts ---")
    for label, cnt in Counter(r["label"] for r in all_labels).most_common():
        print(f"  {label:28s}  {cnt}")


if __name__ == "__main__":
    main()
