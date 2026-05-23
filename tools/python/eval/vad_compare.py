"""Compare fixed-window chunking vs DSP VAD segmentation on a label file.

Implements a Python port of runtime/cpp/src/chunker/vad.cpp so the
comparison can run without building the C++ binary.

Usage:
    python tools/python/eval/vad_compare.py \\
        --labels data/labels/eval_subset.jsonl \\
        [--chunk-sec 5.0] [--max-files N] [--out results.tsv]
"""

import argparse
import collections
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf

_REPO_ROOT = Path(__file__).resolve().parents[3]


def resolve_path(p: str) -> str:
    pp = Path(p)
    return str(pp if pp.is_absolute() else _REPO_ROOT / pp)


def load_labels(path: str) -> List[Dict]:
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# Python VAD — faithful port of vad.cpp
# ---------------------------------------------------------------------------

class VadConfig:
    frame_ms: int        = 20
    hop_ms: int          = 10
    energy_thresh: float = 0.005
    zcr_max: float       = 3000.0
    hangover_frames: int = 8
    min_speech_ms: int   = 200
    min_silence_ms: int  = 100


def run_vad(samples: np.ndarray, sr: int, cfg: Optional[VadConfig] = None) -> List[Dict]:
    """Return list of segment dicts matching VadSegment fields."""
    if cfg is None:
        cfg = VadConfig()

    n = len(samples)
    frame_samp = cfg.frame_ms * sr // 1000
    hop_samp   = cfg.hop_ms   * sr // 1000

    if frame_samp <= 0 or hop_samp <= 0 or n < frame_samp:
        return []

    n_frames = (n - frame_samp) // hop_samp + 1

    # Step 1: raw per-frame classification
    raw_speech = []
    for f in range(n_frames):
        start = f * hop_samp
        frame = samples[start : start + frame_samp]
        rms = math.sqrt(float(np.mean(frame ** 2)))
        # ZCR: count sign changes
        signs = np.sign(frame)
        crossings = int(np.sum(signs[1:] != signs[:-1]))
        zcr = crossings * sr / frame_samp
        raw_speech.append(rms >= cfg.energy_thresh and zcr < cfg.zcr_max)

    # Step 2: hangover
    smooth = [False] * n_frames
    hold = 0
    for f in range(n_frames):
        if raw_speech[f]:
            hold = cfg.hangover_frames + 1
        if hold > 0:
            smooth[f] = True
            hold -= 1

    # Step 3: extract contiguous segments
    segs = []
    f = 0
    while f < n_frames:
        if not smooth[f]:
            f += 1
            continue
        s = f
        while f < n_frames and smooth[f]:
            f += 1
        segs.append([s, f])

    if not segs:
        return []

    # Step 4: merge short silences
    min_sil_frames = max(1, cfg.min_silence_ms * sr // (1000 * hop_samp))
    merged = [list(segs[0])]
    for s, e in segs[1:]:
        if s - merged[-1][1] <= min_sil_frames:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    # Step 5: filter and convert
    min_sp_frames = max(1, cfg.min_speech_ms * sr // (1000 * hop_samp))
    result = []
    for s, e in merged:
        length = e - s
        if length < min_sp_frames:
            continue
        raw_count = sum(1 for ff in range(s, e) if raw_speech[ff])
        start_sec = s * hop_samp / sr
        end_sec   = min((e - 1) * hop_samp + frame_samp, n) / sr
        result.append({
            "start_sec":    start_sec,
            "end_sec":      end_sec,
            "duration_sec": end_sec - start_sec,
            "speech_ratio": raw_count / length if length else 0.0,
            "frame_count":  length,
        })

    return result


# ---------------------------------------------------------------------------
# Comparison logic
# ---------------------------------------------------------------------------

def compare_file(path: str, duration_sec: float, chunk_sec: float,
                 cfg: VadConfig) -> Optional[Dict]:
    """Return per-file comparison stats, or None on load error."""
    abs_path = resolve_path(path)
    try:
        data, sr = sf.read(abs_path, dtype="float32", always_2d=False)
    except Exception as exc:
        print(f"  SKIP {os.path.basename(abs_path)}: {exc}", file=sys.stderr)
        return None

    if data.ndim == 2:
        data = data.mean(axis=1)

    audio_sec = len(data) / sr

    # Fixed chunking: non-overlapping chunks of chunk_sec
    n_fixed = max(1, math.ceil(audio_sec / chunk_sec)) if audio_sec > 0 else 0
    fixed_retained = audio_sec  # fixed chunking keeps all audio

    # VAD
    segs = run_vad(data, sr, cfg)
    vad_retained = sum(s["duration_sec"] for s in segs)

    return {
        "audio_sec":      audio_sec,
        "n_fixed_chunks": n_fixed,
        "n_vad_segments": len(segs),
        "fixed_retained": fixed_retained,
        "vad_retained":   vad_retained,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels",    default="data/labels/eval_subset.jsonl",
                    help="JSONL label file (default: data/labels/eval_subset.jsonl)")
    ap.add_argument("--chunk-sec", type=float, default=5.0,
                    help="Fixed chunk length in seconds (default: 5.0)")
    ap.add_argument("--max-files", type=int, default=None,
                    help="Limit to first N files per label (default: all)")
    ap.add_argument("--out",       default=None,
                    help="Write per-label TSV to this path")
    args = ap.parse_args()

    labels_path = resolve_path(args.labels)
    entries = load_labels(labels_path)
    if not entries:
        print("ERROR: no entries loaded from", labels_path, file=sys.stderr)
        sys.exit(1)

    vad_cfg = VadConfig()

    # Group by label, optionally cap per-label count
    by_label: Dict[str, List[Dict]] = collections.defaultdict(list)
    for e in entries:
        by_label[e.get("label", "unknown")].append(e)
    if args.max_files:
        by_label = {k: v[: args.max_files] for k, v in by_label.items()}

    total_files = sum(len(v) for v in by_label.values())
    print(f"Label file   : {labels_path}")
    print(f"Total files  : {total_files}")
    print(f"Chunk sec    : {args.chunk_sec}")
    print()

    # Per-label accumulators
    label_stats: Dict[str, Dict] = {}
    grand = collections.Counter()

    col_w = 22
    hdr = (f"{'label':<{col_w}}  {'files':>5}  {'fixed_chunks':>12}  "
           f"{'vad_segs':>8}  {'audio_sec':>9}  "
           f"{'vad_retained':>12}  {'removed_pct':>11}")
    print(hdr)
    print("-" * len(hdr))

    for label in sorted(by_label):
        acc: Dict = collections.Counter()
        for entry in by_label[label]:
            dur = entry.get("duration_sec", args.chunk_sec)
            r = compare_file(entry["path"], dur, args.chunk_sec, vad_cfg)
            if r is None:
                continue
            for k, v in r.items():
                acc[k] += v
            acc["n_files"] += 1

        if acc["n_files"] == 0:
            continue

        audio_sec  = acc["audio_sec"]
        vad_ret    = acc["vad_retained"]
        removed    = audio_sec - vad_ret
        removed_pct = 100.0 * removed / audio_sec if audio_sec > 0 else 0.0

        label_stats[label] = dict(acc)
        label_stats[label]["removed_pct"] = removed_pct

        print(
            f"{label:<{col_w}}  {acc['n_files']:>5}  "
            f"{acc['n_fixed_chunks']:>12}  {acc['n_vad_segments']:>8}  "
            f"{audio_sec:>9.1f}  {vad_ret:>12.1f}  {removed_pct:>10.1f}%"
        )

        for k, v in acc.items():
            grand[k] += v

    if not grand:
        print("No files processed.", file=sys.stderr)
        sys.exit(1)

    print("-" * len(hdr))
    total_audio = grand["audio_sec"]
    total_vad   = grand["vad_retained"]
    total_removed_pct = 100.0 * (total_audio - total_vad) / total_audio if total_audio > 0 else 0.0
    print(
        f"{'TOTAL':<{col_w}}  {grand['n_files']:>5}  "
        f"{grand['n_fixed_chunks']:>12}  {grand['n_vad_segments']:>8}  "
        f"{total_audio:>9.1f}  {total_vad:>12.1f}  {total_removed_pct:>10.1f}%"
    )
    print()
    print(f"Total audio      : {total_audio:.1f} s")
    print(f"VAD retained     : {total_vad:.1f} s  ({100.0 - total_removed_pct:.1f}% kept)")
    print(f"Compute saved    : {total_audio - total_vad:.1f} s  ({total_removed_pct:.1f}% removed)")
    print(f"Fixed chunks     : {grand['n_fixed_chunks']}")
    print(f"VAD segments     : {grand['n_vad_segments']}")

    if args.out:
        with open(args.out, "w") as fh:
            fh.write("label\tn_files\tn_fixed_chunks\tn_vad_segments\t"
                     "audio_sec\tvad_retained\tremoved_pct\n")
            for label, s in sorted(label_stats.items()):
                fh.write(
                    f"{label}\t{s['n_files']}\t{s['n_fixed_chunks']}\t"
                    f"{s['n_vad_segments']}\t{s['audio_sec']:.2f}\t"
                    f"{s['vad_retained']:.2f}\t{s['removed_pct']:.2f}\n"
                )
        print(f"\nPer-label TSV    : {args.out}")


if __name__ == "__main__":
    main()
