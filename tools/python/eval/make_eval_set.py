#!/usr/bin/env python3
"""
make_eval_set.py
Build a small labeled eval set from stage-1 datasets.

This script takes clean LibriSpeech chunks and optionally mixes them with
MUSAN/DEMAND noise files to create a diverse labeled evaluation set.

Output labels per chunk:
    clean_speech          - LibriSpeech chunk, no added noise
    speech_in_noise       - clean speech + noise mix
    silence               - synthesized silence
    stationary_noise      - MUSAN noise file chunk
    music                 - MUSAN music file chunk
    clipped_or_distorted  - clean speech clipped programmatically
    low_utility           - very short or extremely low energy chunk

Operational label:
    should_transcribe     - yes / no

Usage:
    python tools/python/eval/make_eval_set.py
    python tools/python/eval/make_eval_set.py --n-clean 100 --snr-db 10
    python tools/python/eval/make_eval_set.py --limit 50  # smoke run

Outputs:
    data/manifests/eval_set.tsv
    data/labels/eval_set_labels.tsv
    data/processed/eval_set/     (WAV chunks)
"""

import argparse
import math
import os
import random
import struct
import sys
import wave
from pathlib import Path
from typing import List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[3]
PROC_DIR  = REPO_ROOT / "data" / "processed"
MAN_DIR   = REPO_ROOT / "data" / "manifests"
LAB_DIR   = REPO_ROOT / "data" / "labels"
EXT_DIR   = REPO_ROOT / "data" / "external"

RANDOM_SEED = 42


# -----------------------------------------------------------------------
# WAV I/O (stdlib only)
# -----------------------------------------------------------------------

def read_wav_mono_f32(path: Path) -> Tuple[List[float], int]:
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
        raise ValueError(f"Unsupported sample width {sampwidth}")

    samples_f = [s * scale for s in samples]
    if n_ch > 1:
        mono = []
        for i in range(0, len(samples_f), n_ch):
            mono.append(sum(samples_f[i:i + n_ch]) / n_ch)
        samples_f = mono
    return samples_f, sr


def write_wav_f32(path: Path, samples: List[float], sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(samples)
    clipped = [max(-32768, min(32767, int(s * 32767))) for s in samples]
    raw = struct.pack(f"<{n}h", *clipped)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(raw)


def rms(samples: List[float]) -> float:
    if not samples:
        return 0.0
    return math.sqrt(sum(s * s for s in samples) / len(samples))


def resample_nearest(samples: List[float], src_sr: int, dst_sr: int) -> List[float]:
    """Very simple nearest-neighbor resample. Good enough for mixing."""
    if src_sr == dst_sr:
        return samples
    ratio = src_sr / dst_sr
    out_len = int(len(samples) / ratio)
    return [samples[min(int(i * ratio), len(samples) - 1)] for i in range(out_len)]


def mix_snr(speech: List[float], noise: List[float], snr_db: float) -> List[float]:
    """Mix speech + noise at a given SNR in dB."""
    rms_s = rms(speech)
    rms_n = rms(noise[:len(speech)])
    if rms_s < 1e-9 or rms_n < 1e-9:
        return speech

    target_noise_rms = rms_s / (10 ** (snr_db / 20.0))
    scale = target_noise_rms / rms_n

    # Tile or trim noise to match speech length
    n = len(speech)
    if len(noise) < n:
        tiled = []
        while len(tiled) < n:
            tiled.extend(noise)
        noise = tiled
    noise = noise[:n]

    mixed = [speech[i] + scale * noise[i] for i in range(n)]
    # Normalize to avoid clipping
    peak = max(abs(s) for s in mixed) or 1.0
    if peak > 0.98:
        mixed = [s / peak * 0.95 for s in mixed]
    return mixed


def apply_clip(samples: List[float], clip_level: float = 0.3) -> List[float]:
    """Hard-clip a signal to simulate clipping distortion."""
    return [max(-1.0, min(1.0, s / clip_level)) * clip_level for s in samples]


# -----------------------------------------------------------------------
# Dataset scanner helpers
# -----------------------------------------------------------------------

def find_wavs(root: Path, limit: Optional[int] = None) -> List[Path]:
    """Recursively find WAV files under root."""
    wavs = sorted(root.rglob("*.wav"))
    if limit:
        wavs = wavs[:limit]
    return wavs


# -----------------------------------------------------------------------
# Eval set builder
# -----------------------------------------------------------------------

LABEL_TRANSCRIBE = {
    "clean_speech":         "yes",
    "speech_in_noise":      "yes",
    "silence":              "no",
    "stationary_noise":     "no",
    "music":                "no",
    "clipped_or_distorted": "no",
    "low_utility":          "no",
}


def build_eval_set(
    n_clean:    int,
    n_noisy:    int,
    n_silence:  int,
    n_noise:    int,
    n_music:    int,
    n_clipped:  int,
    snr_db:     float,
    chunk_sec:  float,
    out_dir:    Path,
) -> List[dict]:
    rng = random.Random(RANDOM_SEED)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []

    # --- Source pools ---
    clean_pool = find_wavs(PROC_DIR / "eval_chunks")
    noise_pool = find_wavs(EXT_DIR / "musan" / "musan" / "noise")
    music_pool = find_wavs(EXT_DIR / "musan" / "musan" / "music")

    target_len = int(chunk_sec * 16000)

    def load_pad_trim(path: Path) -> Tuple[List[float], int]:
        samples, sr = read_wav_mono_f32(path)
        if sr != 16000:
            samples = resample_nearest(samples, sr, 16000)
            sr = 16000
        if len(samples) < target_len:
            samples = samples + [0.0] * (target_len - len(samples))
        return samples[:target_len], sr

    def write_chunk(name: str, samples: List[float], sr: int) -> Path:
        p = out_dir / name
        write_wav_f32(p, samples, sr)
        return p

    idx = 0

    # --- Clean speech ---
    clean_sample = rng.sample(clean_pool, min(n_clean, len(clean_pool))) if clean_pool else []
    for src in clean_sample:
        samples, sr = load_pad_trim(src)
        name = f"chunk_{idx:05d}_clean.wav"
        p = write_chunk(name, samples, sr)
        records.append({
            "path": str(p), "scene_label": "clean_speech",
            "should_transcribe": "yes", "source": str(src),
        })
        idx += 1

    # --- Noisy speech ---
    if clean_sample and noise_pool:
        noisy_sample = rng.sample(clean_pool, min(n_noisy, len(clean_pool)))
        noise_files  = noise_pool * (n_noisy // max(len(noise_pool), 1) + 1)
        for i, src in enumerate(noisy_sample):
            speech, sr = load_pad_trim(src)
            noise, _   = load_pad_trim(noise_files[i % len(noise_pool)])
            mixed = mix_snr(speech, noise, snr_db)
            name = f"chunk_{idx:05d}_noisy.wav"
            p = write_chunk(name, mixed, sr)
            records.append({
                "path": str(p), "scene_label": "speech_in_noise",
                "should_transcribe": "yes", "source": str(src),
            })
            idx += 1

    # --- Silence ---
    for i in range(n_silence):
        silence = [0.0] * target_len
        name = f"chunk_{idx:05d}_silence.wav"
        p = write_chunk(name, silence, 16000)
        records.append({
            "path": str(p), "scene_label": "silence",
            "should_transcribe": "no", "source": "synthetic",
        })
        idx += 1

    # --- Stationary noise ---
    noise_sample = rng.sample(noise_pool, min(n_noise, len(noise_pool))) if noise_pool else []
    for src in noise_sample:
        samples, sr = load_pad_trim(src)
        name = f"chunk_{idx:05d}_noise.wav"
        p = write_chunk(name, samples, sr)
        records.append({
            "path": str(p), "scene_label": "stationary_noise",
            "should_transcribe": "no", "source": str(src),
        })
        idx += 1

    # --- Music ---
    music_sample = rng.sample(music_pool, min(n_music, len(music_pool))) if music_pool else []
    for src in music_sample:
        samples, sr = load_pad_trim(src)
        name = f"chunk_{idx:05d}_music.wav"
        p = write_chunk(name, samples, sr)
        records.append({
            "path": str(p), "scene_label": "music",
            "should_transcribe": "no", "source": str(src),
        })
        idx += 1

    # --- Clipped speech ---
    clipped_sample = rng.sample(clean_pool, min(n_clipped, len(clean_pool))) if clean_pool else []
    for src in clipped_sample:
        samples, sr = load_pad_trim(src)
        clipped = apply_clip(samples, clip_level=0.25)
        name = f"chunk_{idx:05d}_clipped.wav"
        p = write_chunk(name, clipped, sr)
        records.append({
            "path": str(p), "scene_label": "clipped_or_distorted",
            "should_transcribe": "no", "source": str(src),
        })
        idx += 1

    return records


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build labeled eval set from stage-1 datasets.")
    parser.add_argument("--n-clean",   type=int,   default=50,  help="Clean speech chunks")
    parser.add_argument("--n-noisy",   type=int,   default=50,  help="Speech-in-noise chunks")
    parser.add_argument("--n-silence", type=int,   default=20,  help="Silence chunks")
    parser.add_argument("--n-noise",   type=int,   default=30,  help="Stationary noise chunks")
    parser.add_argument("--n-music",   type=int,   default=20,  help="Music chunks")
    parser.add_argument("--n-clipped", type=int,   default=20,  help="Clipped speech chunks")
    parser.add_argument("--snr-db",    type=float, default=10.0, help="SNR for noisy mixes (dB)")
    parser.add_argument("--chunk-sec", type=float, default=5.0,  help="Chunk length in seconds")
    parser.add_argument("--limit",     type=int,   default=None, help="Cap all classes at N (smoke mode)")
    args = parser.parse_args()

    if args.limit:
        cap = args.limit
        args.n_clean   = min(args.n_clean,   cap)
        args.n_noisy   = min(args.n_noisy,   cap)
        args.n_silence = min(args.n_silence, cap)
        args.n_noise   = min(args.n_noise,   cap)
        args.n_music   = min(args.n_music,   cap)
        args.n_clipped = min(args.n_clipped, cap)

    out_dir = PROC_DIR / "eval_set"
    print(f"Building eval set -> {out_dir}")

    records = build_eval_set(
        n_clean   = args.n_clean,
        n_noisy   = args.n_noisy,
        n_silence = args.n_silence,
        n_noise   = args.n_noise,
        n_music   = args.n_music,
        n_clipped = args.n_clipped,
        snr_db    = args.snr_db,
        chunk_sec = args.chunk_sec,
        out_dir   = out_dir,
    )

    if not records:
        print("No records produced. Check that stage-1 datasets are downloaded.")
        sys.exit(1)

    # Write manifest
    man_path = MAN_DIR / "eval_set.tsv"
    man_path.parent.mkdir(parents=True, exist_ok=True)
    with open(man_path, "w") as f:
        keys = list(records[0].keys())
        f.write("\t".join(keys) + "\n")
        for r in records:
            f.write("\t".join(str(r.get(k, "")) for k in keys) + "\n")
    print(f"Manifest: {man_path}  ({len(records)} records)")

    # Write labels
    lab_path = LAB_DIR / "eval_set_labels.tsv"
    lab_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lab_path, "w") as f:
        f.write("path\tscene_label\tshould_transcribe\n")
        for r in records:
            f.write(f"{r['path']}\t{r['scene_label']}\t{r['should_transcribe']}\n")
    print(f"Labels:   {lab_path}")

    # Scene distribution
    from collections import Counter
    dist = Counter(r["scene_label"] for r in records)
    print("\nScene distribution:")
    for label, count in sorted(dist.items()):
        should = LABEL_TRANSCRIBE.get(label, "?")
        print(f"  {label:<28s} {count:4d}  should_transcribe={should}")


if __name__ == "__main__":
    main()
