#!/usr/bin/env python3
"""
build_eval_subset.py
Build a small, reproducible, labeled evaluation subset for gate testing
and ASR smoke benchmarking.

Requirements:
    conda env create -f environment.yml
    conda run -n audio_king python scripts/datasets/build_eval_subset.py --seed 42 --overwrite

Reads manifests from data/manifests/ (run build_manifests.py first).

Output:
    data/processed/eval_subset/          - synthetic WAVs + optional clean symlinks
    data/labels/eval_subset.jsonl        - one record per example with label schema

Label schema:
    label             : clean_speech | speech_in_noise | speech_in_reverb |
                        stationary_noise | music | clipped_or_distorted | low_utility
    should_transcribe : yes | no
    synthetic         : true | false
    corruption_source : source dataset/file (when applicable)
    snr_db            : target SNR in dB (speech_in_noise only)
    rir_id            : RIR file stem (speech_in_reverb only)
    generation_seed   : seed used for deterministic generation
    duration_sec      : output example duration
    sample_rate       : output sample rate (16 kHz for rendered examples)

Target counts (adjustable via CLI):
    clean_speech         100
    speech_in_noise      100
    speech_in_reverb      50
    music                 50
    stationary_noise      50
    clipped_or_distorted  50
    low_utility           50
"""

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[2]
MAN_DIR = REPO_ROOT / "data" / "manifests"
OUT_DIR = REPO_ROOT / "data" / "processed" / "eval_subset"
LABEL_PATH = REPO_ROOT / "data" / "labels" / "eval_subset.jsonl"

TARGET_SR = 16000
CHUNK_SEC = 5

DEFAULT_COUNTS = {
    "clean_speech": 100,
    "speech_in_noise": 100,
    "speech_in_reverb": 50,
    "music": 50,
    "stationary_noise": 50,
    "clipped_or_distorted": 50,
    "low_utility": 50,
}

SNR_MIN, SNR_MAX = 0, 20


def load_mono_16k(path: Path) -> np.ndarray | None:
    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception as exc:
        print(f"  WARN: cannot read {path}: {exc}", file=sys.stderr)
        return None

    if data.ndim == 2:
        data = data.mean(axis=1)

    if sr != TARGET_SR:
        if sr % TARGET_SR == 0:
            data = data[:: (sr // TARGET_SR)]
        elif TARGET_SR % sr == 0:
            data = np.repeat(data, TARGET_SR // sr)
        else:
            n_out = int(len(data) * TARGET_SR / sr)
            idx = (np.arange(n_out) * sr / TARGET_SR).astype(int)
            idx = np.clip(idx, 0, len(data) - 1)
            data = data[idx]

    return data


def save_wav(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, TARGET_SR, subtype="PCM_16")


def trim_or_pad(audio: np.ndarray, target_samples: int) -> np.ndarray:
    if len(audio) >= target_samples:
        return audio[:target_samples]
    reps = (target_samples // len(audio)) + 1
    return np.tile(audio, reps)[:target_samples]


def normalize_manifest_record(rec: dict) -> dict:
    # Backward compatibility for historical schema variants.
    rec["sample_rate"] = rec.get("sample_rate", rec.get("sr"))
    rec["duration_sec"] = rec.get("duration_sec", rec.get("duration"))
    return rec


def load_manifest(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = normalize_manifest_record(json.loads(line))
            if "path" not in rec:
                continue
            rows.append(rec)
    return rows


def prepare_outputs(overwrite: bool, dry_run: bool) -> None:
    if dry_run:
        return

    has_existing = OUT_DIR.exists() and any(OUT_DIR.rglob("*"))
    if has_existing and not overwrite:
        print(
            f"ERROR: output dir already has data: {OUT_DIR}\n"
            "Use --overwrite to remove and regenerate deterministically.",
            file=sys.stderr,
        )
        sys.exit(1)

    if overwrite and OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    if overwrite and LABEL_PATH.exists():
        LABEL_PATH.unlink()


def make_label(
    *,
    out_path: Path,
    label: str,
    should_transcribe: str,
    synthetic: bool,
    source: str,
    source_type: str,
    base_utterance_id: str,
    corruption_source: str,
    snr_db: float | None,
    rir_id: str,
    seed: int,
    duration_sec: float,
    sample_rate: int,
) -> dict:
    return {
        "path": str(out_path),
        "label": label,
        "should_transcribe": should_transcribe,
        "synthetic": synthetic,
        "source": source,
        "source_type": source_type,
        "base_utterance_id": base_utterance_id,
        "corruption_source": corruption_source,
        "snr_db": snr_db,
        "rir_id": rir_id,
        "generation_seed": seed,
        "duration_sec": round(float(duration_sec), 4),
        "sample_rate": int(sample_rate),
    }


def _speech_records(dev_clean_manifest: list[dict], rng: random.Random, n: int) -> list[dict]:
    pool = [r for r in dev_clean_manifest if Path(r["path"]).exists()]
    if len(pool) < n:
        print(
            f"  WARN: only {len(pool)} readable dev-clean utterances, requested {n}",
            file=sys.stderr,
        )
    return rng.sample(pool, min(n, len(pool)))


def mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    speech_rms = np.sqrt(np.mean(speech ** 2)) + 1e-9
    noise_rms = np.sqrt(np.mean(noise ** 2)) + 1e-9
    target_noise_rms = speech_rms / (10 ** (snr_db / 20))
    noise_scaled = noise * (target_noise_rms / noise_rms)
    mixed = speech + noise_scaled
    peak = np.max(np.abs(mixed))
    if peak > 1.0:
        mixed /= peak
    return mixed


def apply_rir(speech: np.ndarray, rir: np.ndarray) -> np.ndarray:
    reverbed = np.convolve(speech, rir)[: len(speech)]
    peak = np.max(np.abs(reverbed))
    if peak > 1e-6:
        reverbed /= peak
    return reverbed


def apply_clipping(audio: np.ndarray, clip_threshold: float) -> np.ndarray:
    clipped = np.clip(audio, -clip_threshold, clip_threshold)
    peak = np.max(np.abs(clipped))
    if peak > 1e-6:
        clipped /= peak
    return clipped


def build_clean_speech(
    speech_pool: list[dict],
    rng: random.Random,
    n: int,
    dry_run: bool,
    seed: int,
    link_clean: bool,
) -> list[dict]:
    selected = _speech_records(speech_pool, rng, n)
    chunk_samples = CHUNK_SEC * TARGET_SR
    labels = []

    for i, rec in enumerate(selected):
        src = Path(rec["path"])
        ext = src.suffix if link_clean else ".wav"
        out_path = OUT_DIR / "clean_speech" / f"clean_speech_{i:04d}{ext}"

        out_duration = float(rec.get("duration_sec") or CHUNK_SEC)
        out_sr = int(rec.get("sample_rate") or TARGET_SR)

        if not dry_run:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if link_clean:
                if out_path.exists() or out_path.is_symlink():
                    out_path.unlink()
                out_path.symlink_to(src)
            else:
                audio = load_mono_16k(src)
                if audio is None:
                    continue
                audio = trim_or_pad(audio, chunk_samples)
                save_wav(out_path, audio)
                out_duration = CHUNK_SEC
                out_sr = TARGET_SR

        labels.append(
            make_label(
                out_path=out_path,
                label="clean_speech",
                should_transcribe="yes",
                synthetic=False,
                source=rec["path"],
                source_type=rec.get("source_type", "clean_speech"),
                base_utterance_id=rec.get("utterance_id", ""),
                corruption_source="",
                snr_db=None,
                rir_id="",
                seed=seed,
                duration_sec=out_duration,
                sample_rate=out_sr,
            )
        )

    return labels


def build_speech_in_noise(
    speech_pool: list[dict],
    noise_pool: list[dict],
    rng: random.Random,
    n: int,
    dry_run: bool,
    seed: int,
) -> list[dict]:
    speech_sel = _speech_records(speech_pool, rng, n)
    noise_pool_ok = [r for r in noise_pool if Path(r["path"]).exists()]
    if not noise_pool_ok:
        print("  WARN: no readable noise examples for speech_in_noise", file=sys.stderr)
        return []

    chunk_samples = CHUNK_SEC * TARGET_SR
    labels = []
    for i, rec in enumerate(speech_sel):
        snr_db = rng.uniform(SNR_MIN, SNR_MAX)
        noise_rec = rng.choice(noise_pool_ok)
        out_path = OUT_DIR / "speech_in_noise" / f"speech_in_noise_{i:04d}.wav"

        if not dry_run:
            speech = load_mono_16k(Path(rec["path"]))
            noise = load_mono_16k(Path(noise_rec["path"]))
            if speech is None or noise is None:
                continue
            speech = trim_or_pad(speech, chunk_samples)
            noise = trim_or_pad(noise, chunk_samples)
            save_wav(out_path, mix_at_snr(speech, noise, snr_db))

        labels.append(
            make_label(
                out_path=out_path,
                label="speech_in_noise",
                should_transcribe="yes",
                synthetic=True,
                source=rec["path"],
                source_type=rec.get("source_type", "clean_speech"),
                base_utterance_id=rec.get("utterance_id", ""),
                corruption_source=noise_rec["path"],
                snr_db=round(snr_db, 1),
                rir_id="",
                seed=seed,
                duration_sec=CHUNK_SEC,
                sample_rate=TARGET_SR,
            )
        )

    return labels


def build_speech_in_reverb(
    speech_pool: list[dict],
    rir_pool: list[dict],
    rng: random.Random,
    n: int,
    dry_run: bool,
    seed: int,
) -> list[dict]:
    speech_sel = _speech_records(speech_pool, rng, n)
    rir_pool_ok = [r for r in rir_pool if Path(r["path"]).exists()]
    if not rir_pool_ok:
        print("  WARN: no readable RIR examples for speech_in_reverb", file=sys.stderr)
        return []

    chunk_samples = CHUNK_SEC * TARGET_SR
    labels = []
    for i, rec in enumerate(speech_sel):
        rir_rec = rng.choice(rir_pool_ok)
        out_path = OUT_DIR / "speech_in_reverb" / f"speech_in_reverb_{i:04d}.wav"

        if not dry_run:
            speech = load_mono_16k(Path(rec["path"]))
            rir = load_mono_16k(Path(rir_rec["path"]))
            if speech is None or rir is None:
                continue
            speech = trim_or_pad(speech, chunk_samples)
            save_wav(out_path, apply_rir(speech, rir))

        labels.append(
            make_label(
                out_path=out_path,
                label="speech_in_reverb",
                should_transcribe="yes",
                synthetic=True,
                source=rec["path"],
                source_type=rec.get("source_type", "clean_speech"),
                base_utterance_id=rec.get("utterance_id", ""),
                corruption_source=rir_rec["path"],
                snr_db=None,
                rir_id=Path(rir_rec["path"]).stem,
                seed=seed,
                duration_sec=CHUNK_SEC,
                sample_rate=TARGET_SR,
            )
        )

    return labels


def build_music(music_pool: list[dict], rng: random.Random, n: int, dry_run: bool, seed: int) -> list[dict]:
    pool = [r for r in music_pool if Path(r["path"]).exists()]
    if not pool:
        print("  WARN: no readable music examples", file=sys.stderr)
        return []

    selected = rng.sample(pool, min(n, len(pool)))
    chunk_samples = CHUNK_SEC * TARGET_SR
    labels = []
    for i, rec in enumerate(selected):
        out_path = OUT_DIR / "music" / f"music_{i:04d}.wav"
        if not dry_run:
            audio = load_mono_16k(Path(rec["path"]))
            if audio is None:
                continue
            save_wav(out_path, trim_or_pad(audio, chunk_samples))

        labels.append(
            make_label(
                out_path=out_path,
                label="music",
                should_transcribe="no",
                synthetic=False,
                source=rec["path"],
                source_type=rec.get("source_type", "music"),
                base_utterance_id="",
                corruption_source="",
                snr_db=None,
                rir_id="",
                seed=seed,
                duration_sec=CHUNK_SEC,
                sample_rate=TARGET_SR,
            )
        )

    return labels


def build_stationary_noise(
    noise_pool: list[dict],
    demand_pool: list[dict],
    rng: random.Random,
    n: int,
    dry_run: bool,
    seed: int,
) -> list[dict]:
    pool = [r for r in demand_pool if Path(r["path"]).exists()]
    if len(pool) < n:
        pool.extend([r for r in noise_pool if Path(r["path"]).exists()])
    if not pool:
        print("  WARN: no readable stationary noise examples", file=sys.stderr)
        return []

    selected = rng.sample(pool, min(n, len(pool)))
    chunk_samples = CHUNK_SEC * TARGET_SR
    labels = []
    for i, rec in enumerate(selected):
        out_path = OUT_DIR / "stationary_noise" / f"stationary_noise_{i:04d}.wav"
        if not dry_run:
            audio = load_mono_16k(Path(rec["path"]))
            if audio is None:
                continue
            save_wav(out_path, trim_or_pad(audio, chunk_samples))

        labels.append(
            make_label(
                out_path=out_path,
                label="stationary_noise",
                should_transcribe="no",
                synthetic=False,
                source=rec["path"],
                source_type=rec.get("source_type", "stationary_noise"),
                base_utterance_id="",
                corruption_source="",
                snr_db=None,
                rir_id="",
                seed=seed,
                duration_sec=CHUNK_SEC,
                sample_rate=TARGET_SR,
            )
        )

    return labels


def build_clipped(
    speech_pool: list[dict],
    rng: random.Random,
    n: int,
    dry_run: bool,
    seed: int,
) -> list[dict]:
    selected = _speech_records(speech_pool, rng, n)
    chunk_samples = CHUNK_SEC * TARGET_SR
    labels = []
    for i, rec in enumerate(selected):
        clip_thresh = rng.uniform(0.1, 0.4)
        out_path = OUT_DIR / "clipped_or_distorted" / f"clipped_{i:04d}.wav"
        if not dry_run:
            audio = load_mono_16k(Path(rec["path"]))
            if audio is None:
                continue
            audio = trim_or_pad(audio, chunk_samples)
            save_wav(out_path, apply_clipping(audio, clip_thresh))

        labels.append(
            make_label(
                out_path=out_path,
                label="clipped_or_distorted",
                should_transcribe="no",
                synthetic=True,
                source=rec["path"],
                source_type=rec.get("source_type", "clean_speech"),
                base_utterance_id=rec.get("utterance_id", ""),
                corruption_source=f"hard_clip@{clip_thresh:.2f}",
                snr_db=None,
                rir_id="",
                seed=seed,
                duration_sec=CHUNK_SEC,
                sample_rate=TARGET_SR,
            )
        )

    return labels


def build_low_utility(
    speech_pool: list[dict],
    rng: random.Random,
    n: int,
    dry_run: bool,
    seed: int,
) -> list[dict]:
    pool = [r for r in speech_pool if Path(r["path"]).exists()]
    if not pool:
        print("  WARN: no readable speech examples for low_utility", file=sys.stderr)
        return []

    selected = rng.sample(pool, min(n, len(pool)))
    chunk_samples = CHUNK_SEC * TARGET_SR
    labels = []
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

        labels.append(
            make_label(
                out_path=out_path,
                label="low_utility",
                should_transcribe="no",
                synthetic=True,
                source=rec["path"],
                source_type=rec.get("source_type", "clean_speech"),
                base_utterance_id=rec.get("utterance_id", ""),
                corruption_source="silence_pad+attenuation",
                snr_db=None,
                rir_id="",
                seed=seed,
                duration_sec=CHUNK_SEC,
                sample_rate=TARGET_SR,
            )
        )

    return labels


def parse_counts(raw: list[str]) -> dict[str, int]:
    counts = dict(DEFAULT_COUNTS)
    for item in raw:
        k, _, v = item.partition("=")
        if k in counts and v:
            counts[k] = int(v)
        else:
            print(f"  WARN: unknown or invalid --counts token '{item}', ignoring", file=sys.stderr)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build reproducible eval subset.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--counts",
        nargs="+",
        default=[],
        metavar="CLASS=N",
        help="Override per-class counts, e.g. clean_speech=50",
    )
    parser.add_argument("--dry-run", action="store_true", help="Plan only; no writes")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing eval subset outputs before generation",
    )
    parser.add_argument(
        "--link-clean",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Symlink clean_speech examples instead of rendering WAV copies (default: true)",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    counts = parse_counts(args.counts)

    print(f"Seed: {args.seed}")
    print(f"Dry run: {args.dry_run}")
    print(f"Overwrite: {args.overwrite}")
    print(f"Link clean: {args.link_clean}")
    print(f"Counts: {counts}")
    print()

    prepare_outputs(overwrite=args.overwrite, dry_run=args.dry_run)

    dev_clean = load_manifest(MAN_DIR / "librispeech_dev_clean.jsonl")
    musan_noise = load_manifest(MAN_DIR / "musan_noise.jsonl")
    musan_music = load_manifest(MAN_DIR / "musan_music.jsonl")
    rirs = load_manifest(MAN_DIR / "rirs.jsonl")
    demand = load_manifest(MAN_DIR / "demand_16k.jsonl")

    missing = []
    if not dev_clean:
        missing.append("librispeech_dev_clean.jsonl")
    if not musan_noise:
        missing.append("musan_noise.jsonl")
    if not musan_music:
        missing.append("musan_music.jsonl")
    if not rirs:
        missing.append("rirs.jsonl")
    if not demand:
        missing.append("demand_16k.jsonl")

    if missing:
        print(f"ERROR: missing manifests: {missing}", file=sys.stderr)
        print("Run build_manifests.py first.", file=sys.stderr)
        sys.exit(1)

    print(f"dev-clean:    {len(dev_clean):5d} records")
    print(f"musan/noise:  {len(musan_noise):5d} records")
    print(f"musan/music:  {len(musan_music):5d} records")
    print(f"rirs:         {len(rirs):5d} records")
    print(f"demand_16k:   {len(demand):5d} records")
    print()

    sim_rirs = [r for r in rirs if r.get("rir_type") == "simulated"]

    all_labels: list[dict] = []
    print("Building clean_speech ...")
    all_labels += build_clean_speech(dev_clean, rng, counts["clean_speech"], args.dry_run, args.seed, args.link_clean)

    print("Building speech_in_noise ...")
    all_labels += build_speech_in_noise(dev_clean, musan_noise + demand, rng, counts["speech_in_noise"], args.dry_run, args.seed)

    print("Building speech_in_reverb ...")
    all_labels += build_speech_in_reverb(dev_clean, sim_rirs, rng, counts["speech_in_reverb"], args.dry_run, args.seed)

    print("Building music ...")
    all_labels += build_music(musan_music, rng, counts["music"], args.dry_run, args.seed)

    print("Building stationary_noise ...")
    all_labels += build_stationary_noise(musan_noise, demand, rng, counts["stationary_noise"], args.dry_run, args.seed)

    print("Building clipped_or_distorted ...")
    all_labels += build_clipped(dev_clean, rng, counts["clipped_or_distorted"], args.dry_run, args.seed)

    print("Building low_utility ...")
    all_labels += build_low_utility(dev_clean, rng, counts["low_utility"], args.dry_run, args.seed)

    if not args.dry_run:
        LABEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LABEL_PATH, "w") as f:
            for rec in all_labels:
                f.write(json.dumps(rec) + "\n")
        print(f"\nLabels written: {LABEL_PATH}  ({len(all_labels)} records)")
        print(f"Audio written:  {OUT_DIR}/")
    else:
        print(f"\nDry run complete. Would generate {len(all_labels)} examples.")

    from collections import Counter

    print("\n--- Class counts ---")
    for label, cnt in Counter(r["label"] for r in all_labels).most_common():
        print(f"  {label:28s}  {cnt}")


if __name__ == "__main__":
    main()
