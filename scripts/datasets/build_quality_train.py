#!/usr/bin/env python3
"""
build_quality_train.py
Build labeled train/validation/test data for the learned quality predictor.

The authoritative --all-splits mode partitions shared music, noise, and RIR
sources before rendering, excludes legacy eval_subset inputs, validates overlap,
and writes labels only after the rendered splits pass validation.

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
    python scripts/datasets/build_quality_train.py --all-splits --overwrite
    python scripts/datasets/build_quality_train.py [--overwrite] [--dry-run]
    python scripts/datasets/build_quality_train.py --counts music=100 --dry-run
"""

import argparse
import json
import os
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np

# Shared utilities from build_eval_subset live in the same directory.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from dataset_identity import (    # noqa: E402
    assert_no_forbidden_overlap,
    build_overlap_report,
    canonical_source_identity,
    deterministic_partition,
    input_source_identities,
    print_overlap_matrix,
)
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
LABEL_DIR  = REPO_ROOT / "data" / "labels"
PROC_DIR   = REPO_ROOT / "data" / "processed"
EVAL_LABEL = REPO_ROOT / "data" / "labels" / "eval_subset.jsonl"

# Defaults preserved for backward compatibility; main() reassigns these via
# the global statement so that builder functions see the correct output dir.
OUT_DIR    = PROC_DIR  / "quality_train"
LABEL_PATH = LABEL_DIR / "quality_train.jsonl"

# Maps --speech-split to manifest filename and derived output stem.
SPLIT_TO_MANIFEST: dict[str, str] = {
    "train-clean-100": "librispeech_train_clean_100.jsonl",
    "dev-clean":       "librispeech_dev_clean.jsonl",
    "test-clean":      "librispeech_test_clean.jsonl",
}
SPLIT_TO_OUTPUT: dict[str, str] = {
    "train-clean-100": "quality_train",
    "dev-clean":       "quality_val",
    "test-clean":      "quality_test",
}

DEFAULT_COUNTS: dict[str, int] = {
    "clean_speech":         400,
    "speech_in_noise":      400,
    "speech_in_reverb":     200,
    "music":                200,
    "stationary_noise":     150,
    "clipped_or_distorted": 200,
    "low_utility":          150,
}

AUTHORITATIVE_SPLITS: dict[str, dict[str, object]] = {
    "train": {
        "speech_split": "train-clean-100",
        "output_stem": "quality_train",
        "seed_offset": 0,
    },
    "validation": {
        "speech_split": "dev-clean",
        "output_stem": "quality_val",
        "seed_offset": 1,
    },
    "test": {
        "speech_split": "test-clean",
        "output_stem": "quality_test",
        "seed_offset": 2,
    },
}


def load_excluded_sources(paths) -> set:
    """Canonical input sources in any label file — exclude from pools.

    Accepts a single Path/str or a list of paths.  Non-existent paths are
    silently skipped.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    excluded = set()
    for path in paths:
        path = Path(path)
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    excluded.update(input_source_identities(rec, REPO_ROOT))
    return excluded


def _sample(pool: list, rng: random.Random, n: int, strict: bool = False) -> list:
    avail = [r for r in pool if Path(r["path"]).exists()]
    if len(avail) < n:
        message = f"only {len(avail)} readable records, requested {n}"
        if strict:
            raise RuntimeError(message)
        print(f"  WARN: {message}", file=sys.stderr)
    return rng.sample(avail, min(n, len(avail)))


def build_clean_speech(
    pool: list, rng: random.Random, n: int, seed: int, dry_run: bool,
    strict: bool = False,
) -> list:
    selected = _sample(pool, rng, n, strict)
    labels = []
    for i, rec in enumerate(selected):
        src = Path(rec["path"])
        out_path = OUT_DIR / "clean_speech" / f"clean_speech_{i:04d}{src.suffix}"
        if not dry_run:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if out_path.exists() or out_path.is_symlink():
                out_path.unlink()
            out_path.symlink_to(os.path.relpath(src, out_path.parent))
        labels.append(make_label(
            out_path=out_path, label="clean_speech", should_transcribe="yes",
            synthetic=False, source=rec["path"], source_type="clean_speech",
            base_utterance_id=rec.get("utterance_id", ""),
            corruption_source="", snr_db=None, rir_id="", seed=seed,
            duration_sec=float(rec.get("duration_sec") or CHUNK_SEC),
            sample_rate=int(rec.get("sample_rate") or TARGET_SR),
            generation_params={"mode": "source_file"},
        ))
    return labels


def build_speech_in_noise(
    pool: list, noise_pool: list, rng: random.Random, n: int, seed: int,
    dry_run: bool, strict: bool = False,
) -> list:
    selected = _sample(pool, rng, n, strict)
    noise_ok = [r for r in noise_pool if Path(r["path"]).exists()]
    if not noise_ok:
        if strict:
            raise RuntimeError("no readable noise for speech_in_noise")
        print("  WARN: no readable noise for speech_in_noise", file=sys.stderr)
        return []
    labels = []
    chunk_samples = int(CHUNK_SEC * TARGET_SR)
    for i, rec in enumerate(selected):
        # Quantize before rendering so the persisted value is the exact
        # transformation parameter used to create the audio.
        snr_db = round(rng.uniform(SNR_MIN, SNR_MAX), 6)
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
            snr_db=snr_db, rir_id="", seed=seed,
            duration_sec=CHUNK_SEC, sample_rate=TARGET_SR,
            generation_params={"mix": "rms_snr", "target_sec": CHUNK_SEC},
        ))
    return labels


def build_speech_in_reverb(
    pool: list, rir_pool: list, rng: random.Random, n: int, seed: int,
    dry_run: bool, strict: bool = False,
) -> list:
    selected = _sample(pool, rng, n, strict)
    rir_ok = [r for r in rir_pool if Path(r["path"]).exists()]
    if not rir_ok:
        if strict:
            raise RuntimeError("no readable RIRs for speech_in_reverb")
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
            generation_params={"convolution": "full_truncated", "target_sec": CHUNK_SEC},
        ))
    return labels


def build_music(
    pool: list, rng: random.Random, n: int, seed: int, dry_run: bool,
    excluded: set, strict: bool = False,
) -> list:
    avail = [
        r for r in pool
        if Path(r["path"]).exists()
        and canonical_source_identity(r["path"], REPO_ROOT) not in excluded
    ]
    if len(avail) < n:
        message = f"only {len(avail)} music records after exclusion, requested {n}"
        if strict:
            raise RuntimeError(message)
        print(f"  WARN: {message}", file=sys.stderr)
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
            generation_params={"mode": "trim_or_repeat", "target_sec": CHUNK_SEC},
        ))
    return labels


def build_stationary_noise(
    noise_pool: list, demand_pool: list,
    rng: random.Random, n: int, seed: int, dry_run: bool, excluded: set,
    strict: bool = False,
) -> list:
    pool = (
        [r for r in demand_pool if Path(r["path"]).exists()
         and canonical_source_identity(r["path"], REPO_ROOT) not in excluded] +
        [r for r in noise_pool if Path(r["path"]).exists()
         and canonical_source_identity(r["path"], REPO_ROOT) not in excluded]
    )
    if len(pool) < n:
        message = f"only {len(pool)} noise records after exclusion, requested {n}"
        if strict:
            raise RuntimeError(message)
        print(f"  WARN: {message}", file=sys.stderr)
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
            generation_params={"mode": "trim_or_repeat", "target_sec": CHUNK_SEC},
        ))
    return labels


def build_clipped(
    pool: list, rng: random.Random, n: int, seed: int, dry_run: bool,
    strict: bool = False,
) -> list:
    selected = _sample(pool, rng, n, strict)
    labels = []
    chunk_samples = int(CHUNK_SEC * TARGET_SR)
    for i, rec in enumerate(selected):
        # Quantize before rendering for the same provenance guarantee as SNR.
        clip_thresh = round(rng.uniform(0.1, 0.4), 6)
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
            generation_params={
                "clip_threshold": clip_thresh,
                "target_sec": CHUNK_SEC,
            },
        ))
    return labels


def build_low_utility(
    pool: list, rng: random.Random, n: int, seed: int, dry_run: bool,
    strict: bool = False,
) -> list:
    selected = _sample(pool, rng, n, strict)
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
            generation_params={
                "gain": 0.05,
                "keep_sec": 0.5,
                "source_region": "tail",
                "target_sec": CHUNK_SEC,
            },
        ))
    return labels


def _parse_counts(tokens: list[str]) -> dict[str, int]:
    counts = dict(DEFAULT_COUNTS)
    for item in tokens:
        key, separator, value = item.partition("=")
        if not separator or key not in counts:
            raise ValueError(f"unknown --counts token: {item}")
        count = int(value)
        if count < 0:
            raise ValueError(f"negative --counts value: {item}")
        counts[key] = count
    return counts


def _filter_excluded(records: list[dict], excluded: set[str]) -> list[dict]:
    return [
        record for record in records
        if canonical_source_identity(record["path"], REPO_ROOT) not in excluded
    ]


def _require_manifest(name: str) -> list[dict]:
    path = MAN_DIR / name
    records = load_manifest(path)
    if not records:
        raise RuntimeError(f"required manifest is missing or empty: {path}")
    return records


def _build_authoritative_split(
    split_name: str,
    speech_pool: list[dict],
    music_pool: list[dict],
    noise_pool: list[dict],
    rir_pool: list[dict],
    counts: dict[str, int],
    seed: int,
    dry_run: bool,
) -> list[dict]:
    global OUT_DIR, LABEL_PATH

    output_stem = str(AUTHORITATIVE_SPLITS[split_name]["output_stem"])
    OUT_DIR = PROC_DIR / output_stem
    LABEL_PATH = LABEL_DIR / f"{output_stem}.jsonl"
    rng = random.Random(seed)

    labels: list[dict] = []
    labels += build_clean_speech(
        speech_pool, rng, counts["clean_speech"], seed, dry_run, strict=True,
    )
    labels += build_speech_in_noise(
        speech_pool, noise_pool, rng, counts["speech_in_noise"], seed,
        dry_run, strict=True,
    )
    labels += build_speech_in_reverb(
        speech_pool, rir_pool, rng, counts["speech_in_reverb"], seed,
        dry_run, strict=True,
    )
    labels += build_music(
        music_pool, rng, counts["music"], seed, dry_run, set(), strict=True,
    )
    labels += build_stationary_noise(
        noise_pool, [], rng, counts["stationary_noise"], seed, dry_run,
        set(), strict=True,
    )
    labels += build_clipped(
        speech_pool, rng, counts["clipped_or_distorted"], seed,
        dry_run, strict=True,
    )
    labels += build_low_utility(
        speech_pool, rng, counts["low_utility"], seed, dry_run, strict=True,
    )
    return labels


def _assert_expected_counts(
    split_name: str,
    records: list[dict],
    counts: dict[str, int],
) -> None:
    actual = Counter(record["label"] for record in records)
    if len(records) != sum(counts.values()) or actual != Counter(counts):
        raise RuntimeError(
            f"{split_name} row-count mismatch: expected={counts}, actual={dict(actual)}"
        )


def _assert_portable_records(splits: dict[str, list[dict]]) -> None:
    for split_name, records in splits.items():
        encoded = "\n".join(json.dumps(record, sort_keys=True) for record in records)
        if "/home/apr" in encoded:
            raise RuntimeError(f"machine-specific path found in {split_name} labels")


def _write_labels(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _run_all_splits(
    counts: dict[str, int],
    seed: int,
    dry_run: bool,
    overwrite: bool,
) -> None:
    split_names = list(AUTHORITATIVE_SPLITS)
    output_paths = {
        name: {
            "labels": LABEL_DIR / f"{spec['output_stem']}.jsonl",
            "audio": PROC_DIR / str(spec["output_stem"]),
        }
        for name, spec in AUTHORITATIVE_SPLITS.items()
    }

    print("Planned authoritative outputs:")
    for name in split_names:
        print(f"  {name:10s} labels={output_paths[name]['labels']}")
        print(f"  {'':10s} audio ={output_paths[name]['audio']}")
    print(f"Seed: {seed}")
    print(f"Counts per split: {counts}")
    print(f"Dry run: {dry_run}")
    print()

    eval_excluded = load_excluded_sources(EVAL_LABEL)
    speech_pools: dict[str, list[dict]] = {}
    for name, spec in AUTHORITATIVE_SPLITS.items():
        manifest = SPLIT_TO_MANIFEST[str(spec["speech_split"])]
        speech_pools[name] = _filter_excluded(
            _require_manifest(manifest), eval_excluded,
        )

    music = _filter_excluded(_require_manifest("musan_music.jsonl"), eval_excluded)
    noise = _filter_excluded(
        _require_manifest("musan_noise.jsonl") + _require_manifest("demand_16k.jsonl"),
        eval_excluded,
    )
    rirs = _filter_excluded(
        [
            record for record in _require_manifest("rirs.jsonl")
            if record.get("rir_type") == "simulated"
        ],
        eval_excluded,
    )
    if not rirs:
        raise RuntimeError("no simulated RIRs remain after exclusions")

    identity = lambda record: canonical_source_identity(record["path"], REPO_ROOT)
    music_splits = deterministic_partition(music, split_names, seed + 1000, identity)
    noise_splits = deterministic_partition(noise, split_names, seed + 2000, identity)
    rir_splits = deterministic_partition(rirs, split_names, seed + 3000, identity)

    print(f"Legacy eval exclusions: {len(eval_excluded)} canonical inputs")
    for name in split_names:
        print(
            f"  {name:10s} speech={len(speech_pools[name])} "
            f"music={len(music_splits[name])} noise={len(noise_splits[name])} "
            f"rirs={len(rir_splits[name])}"
        )
    print()

    planned: dict[str, list[dict]] = {}
    for name, spec in AUTHORITATIVE_SPLITS.items():
        split_seed = seed + int(spec["seed_offset"])
        planned[name] = _build_authoritative_split(
            name,
            speech_pools[name],
            music_splits[name],
            noise_splits[name],
            rir_splits[name],
            counts,
            split_seed,
            dry_run=True,
        )
        _assert_expected_counts(name, planned[name], counts)

    _assert_portable_records(planned)
    planned_overlap = build_overlap_report(planned, REPO_ROOT)
    print("Planned overlap matrix:")
    print_overlap_matrix(planned_overlap)
    assert_no_forbidden_overlap(planned_overlap)

    if dry_run:
        print("\nDry run passed; no files were written.")
        return

    occupied = []
    for paths in output_paths.values():
        label_path = paths["labels"]
        audio_dir = paths["audio"]
        if label_path.exists() or (audio_dir.exists() and any(audio_dir.rglob("*"))):
            occupied.extend([str(label_path), str(audio_dir)])
    if occupied and not overwrite:
        raise RuntimeError(
            "authoritative outputs already exist; use --overwrite: " + ", ".join(occupied)
        )

    if overwrite:
        for paths in output_paths.values():
            audio_dir = paths["audio"]
            label_path = paths["labels"]
            if audio_dir.exists():
                shutil.rmtree(audio_dir)
            if label_path.exists():
                label_path.unlink()

    rendered: dict[str, list[dict]] = {}
    for name, spec in AUTHORITATIVE_SPLITS.items():
        print(f"Rendering {name} ...", flush=True)
        split_seed = seed + int(spec["seed_offset"])
        rendered[name] = _build_authoritative_split(
            name,
            speech_pools[name],
            music_splits[name],
            noise_splits[name],
            rir_splits[name],
            counts,
            split_seed,
            dry_run=False,
        )
        _assert_expected_counts(name, rendered[name], counts)

    _assert_portable_records(rendered)
    rendered_overlap = build_overlap_report(
        rendered, REPO_ROOT, include_content_hashes=True,
    )
    print("\nRendered overlap matrix:")
    print_overlap_matrix(rendered_overlap)
    assert_no_forbidden_overlap(rendered_overlap)

    for name in split_names:
        _write_labels(output_paths[name]["labels"], rendered[name])
    print("\nAuthoritative labels written after overlap validation.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--all-splits", action="store_true",
        help="Build authoritative disjoint train/validation/test outputs",
    )
    ap.add_argument(
        "--speech-split", default="test-clean",
        choices=list(SPLIT_TO_MANIFEST),
        help="LibriSpeech split for speech-derived labels (default: test-clean)",
    )
    ap.add_argument("--output-labels", default=None, metavar="PATH",
                    help="Override label output path (default: derived from --speech-split)")
    ap.add_argument("--output-dir",    default=None, metavar="DIR",
                    help="Override processed-audio output dir (default: derived from --speech-split)")
    ap.add_argument("--seed",      type=int,  default=123,
                    help="RNG seed (default: 123; eval_subset uses 42)")
    ap.add_argument("--counts",    nargs="+", default=[], metavar="CLASS=N",
                    help="Override per-class counts, e.g. music=100")
    ap.add_argument("--overwrite", action="store_true",
                    help="Delete existing output before regenerating")
    ap.add_argument("--dry-run",   action="store_true",
                    help="Plan only; no writes")
    args = ap.parse_args()

    try:
        counts = _parse_counts(args.counts)
    except (TypeError, ValueError) as exc:
        ap.error(str(exc))

    if args.all_splits:
        if args.output_labels or args.output_dir:
            ap.error("--output-labels/--output-dir cannot be used with --all-splits")
        _run_all_splits(counts, args.seed, args.dry_run, args.overwrite)
        return

    # Resolve output paths from split name unless explicitly overridden.
    split_key  = SPLIT_TO_OUTPUT[args.speech_split]
    out_dir    = Path(args.output_dir)    if args.output_dir    else PROC_DIR  / split_key
    label_path = Path(args.output_labels) if args.output_labels else LABEL_DIR / f"{split_key}.jsonl"

    # Update module-level OUT_DIR / LABEL_PATH so builder functions (which
    # reference these names at call time) see the correct output directory.
    global OUT_DIR, LABEL_PATH
    OUT_DIR    = out_dir
    LABEL_PATH = label_path

    print(f"Speech split:  {args.speech_split}")
    print(f"Output labels: {label_path}")
    print(f"Output dir:    {out_dir}")
    print(f"Seed:          {args.seed}")
    print(f"Dry run:       {args.dry_run}")
    print(f"Overwrite:     {args.overwrite}")
    print(f"Counts:        {counts}")
    print(f"Total target:  {sum(counts.values())}")
    print()

    if not args.dry_run:
        if out_dir.exists() and any(out_dir.rglob("*")):
            if not args.overwrite:
                print(
                    f"ERROR: {out_dir} already has data. Use --overwrite.",
                    file=sys.stderr,
                )
                sys.exit(1)
            shutil.rmtree(out_dir)
        if label_path.exists() and args.overwrite:
            label_path.unlink()

    # Build exclusion set: always include eval_subset; also include any other
    # quality split label files already on disk to prevent cross-split
    # music / noise source leakage.
    all_quality_keys = list(SPLIT_TO_OUTPUT.values())
    exclude_paths = [EVAL_LABEL] + [
        LABEL_DIR / f"{k}.jsonl"
        for k in all_quality_keys
        if k != split_key
    ]
    excluded = load_excluded_sources(exclude_paths)
    n_excl_files = sum(1 for p in exclude_paths if Path(p).exists())
    print(f"Excluded sources ({n_excl_files} label files checked): {len(excluded)} paths")
    print()

    rng = random.Random(args.seed)

    # Load speech manifest for the requested split.  Fail with a helpful
    # message when train-clean-100 is not yet downloaded.
    manifest_name = SPLIT_TO_MANIFEST[args.speech_split]
    manifest_path = MAN_DIR / manifest_name
    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        if args.speech_split == "train-clean-100":
            data_path = REPO_ROOT / "data" / "raw" / "librispeech" / "train-clean-100"
            print(f"       Expected data at:  {data_path}", file=sys.stderr)
            print(f"       Download from:     https://www.openslr.org/12/", file=sys.stderr)
            print(f"       Then rebuild manifests and re-run this script.", file=sys.stderr)
            print(f"       Fallback options:", file=sys.stderr)
            print(f"         --speech-split dev-clean   (val set, present)", file=sys.stderr)
            print(f"         --speech-split test-clean  (test set fallback, present)", file=sys.stderr)
        sys.exit(1)

    speech_pool = load_manifest(manifest_path)
    musan_noise = load_manifest(MAN_DIR / "musan_noise.jsonl")
    musan_music = load_manifest(MAN_DIR / "musan_music.jsonl")
    rirs        = load_manifest(MAN_DIR / "rirs.jsonl")
    demand      = load_manifest(MAN_DIR / "demand_16k.jsonl")

    missing = []
    if not speech_pool: missing.append(manifest_name)
    if not musan_noise: missing.append("musan_noise.jsonl")
    if not musan_music: missing.append("musan_music.jsonl")
    if not rirs:        missing.append("rirs.jsonl")
    if not demand:      missing.append("demand_16k.jsonl")
    if missing:
        print(f"ERROR: missing manifests: {missing}", file=sys.stderr)
        sys.exit(1)

    sim_rirs = [r for r in rirs if r.get("rir_type") == "simulated"]

    print(f"{args.speech_split}:   {len(speech_pool):5d} records")
    print(f"musan/noise:      {len(musan_noise):5d} records")
    print(f"musan/music:      {len(musan_music):5d} records")
    print(f"simulated RIRs:   {len(sim_rirs):5d} records")
    print(f"demand_16k:       {len(demand):5d} records")
    print()

    seed = args.seed
    all_labels: list = []

    print("Building clean_speech ...")
    all_labels += build_clean_speech(speech_pool, rng, counts["clean_speech"], seed, args.dry_run)

    print("Building speech_in_noise ...")
    all_labels += build_speech_in_noise(
        speech_pool, musan_noise + demand, rng, counts["speech_in_noise"], seed, args.dry_run)

    print("Building speech_in_reverb ...")
    all_labels += build_speech_in_reverb(
        speech_pool, sim_rirs, rng, counts["speech_in_reverb"], seed, args.dry_run)

    print("Building music ...")
    all_labels += build_music(musan_music, rng, counts["music"], seed, args.dry_run, excluded)

    print("Building stationary_noise ...")
    all_labels += build_stationary_noise(
        musan_noise, demand, rng, counts["stationary_noise"], seed, args.dry_run, excluded)

    print("Building clipped_or_distorted ...")
    all_labels += build_clipped(speech_pool, rng, counts["clipped_or_distorted"], seed, args.dry_run)

    print("Building low_utility ...")
    all_labels += build_low_utility(speech_pool, rng, counts["low_utility"], seed, args.dry_run)

    if not args.dry_run:
        label_path.parent.mkdir(parents=True, exist_ok=True)
        with open(label_path, "w") as f:
            for rec in all_labels:
                f.write(json.dumps(rec) + "\n")
        print(f"\nLabels written: {label_path}  ({len(all_labels)} records)")
        print(f"Audio written:  {out_dir}/")
    else:
        print(f"\nDry run complete. Would generate {len(all_labels)} examples.")

    print("\n--- Class counts ---")
    for label, cnt in Counter(r["label"] for r in all_labels).most_common():
        print(f"  {label:28s}  {cnt}")


if __name__ == "__main__":
    main()
