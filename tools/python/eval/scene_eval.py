#!/usr/bin/env python3
"""
scene_eval.py
Analyze scene classification output from bench JSON files.

Reads JSON files produced by audio_pipeline (--bench-json), which now include
a "scene" field per chunk added in M5. Reports:
  - Scene distribution per file
  - Scene vs gate decision cross-tabulation
  - Per-scene feature statistics (rms, flatness, centroid, band energies)
  - ASR inference rate per scene (how often ASR ran vs was suppressed)
  - Optionally: per-scene WER if --ref is provided

Usage:
    python tools/python/eval/scene_eval.py benchmarks/results/run.json
    python tools/python/eval/scene_eval.py run.json --ref data/refs.tsv
    python tools/python/eval/scene_eval.py a.json b.json --csv
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional

SCENES    = ["SPEECH", "MIXED", "MUSIC", "NOISE", "SILENCE", "UNKNOWN"]
DECISIONS = ["PASS", "BORDERLINE", "FAIL"]


def load_bench(path: str):
    with open(path) as f:
        data = json.load(f)
    chunks = data.get("chunks", [])
    # Normalize centroid key
    for c in chunks:
        if "centroid_hz" not in c and "centroid" in c:
            c["centroid_hz"] = c["centroid"]
        if "active_frac" not in c and "active_frame_frac" in c:
            c["active_frac"] = c["active_frame_frac"]
    return chunks, data.get("config", {}), data.get("summary", {})


def pct(vals: list, p: float) -> float:
    if not vals:
        return 0.0
    sv = sorted(vals)
    idx = (len(sv) - 1) * p
    lo, hi = int(idx), min(int(idx) + 1, len(sv) - 1)
    return sv[lo] + (sv[hi] - sv[lo]) * (idx - lo)


def cell(v: Optional[float], fmt: str = ".3f") -> str:
    if v is None:
        return "  n/a"
    return format(v, fmt)


def _wer_for_chunks(chunks: list, ref_map: Dict[int, str]) -> Optional[float]:
    total_ref = 0
    total_err = 0
    for c in chunks:
        ref = ref_map.get(c.get("idx", -1), "")
        if not ref:
            continue
        hyp = c.get("transcript", "") or ""
        words_r = ref.lower().split()
        words_h = hyp.lower().split()
        n, m = len(words_r), len(words_h)
        dp = list(range(n + 1))
        for j in range(1, m + 1):
            prev = dp[:]
            dp[0] = j
            for i in range(1, n + 1):
                dp[i] = prev[i - 1] if words_r[i - 1] == words_h[j - 1] \
                    else 1 + min(prev[i - 1], prev[i], dp[i - 1])
        total_ref += n
        total_err += dp[n]
    return total_err / total_ref if total_ref > 0 else None


def analyze(path: str, ref_map: Dict[int, str], csv_mode: bool):
    chunks, config, summary = load_bench(path)
    total = len(chunks)
    if total == 0:
        print(f"WARNING: no chunks in {path}", file=sys.stderr)
        return

    # Group by scene
    by_scene: Dict[str, List[dict]] = defaultdict(list)
    for c in chunks:
        label = c.get("scene", "UNKNOWN")
        by_scene[label].append(c)

    # Group by (scene, decision) for cross-tab
    cross: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for c in chunks:
        s = c.get("scene", "UNKNOWN")
        d = c.get("decision", "UNKNOWN")
        cross[s][d] += 1

    if csv_mode:
        _csv_output(path, chunks, by_scene, cross, ref_map, config, summary)
    else:
        _table_output(path, chunks, by_scene, cross, ref_map, config, summary)


def _table_output(path, chunks, by_scene, cross, ref_map, config, summary):
    total = len(chunks)
    print(f"\n{'=' * 72}")
    print(f"File    : {os.path.basename(path)}")
    print(f"Model   : {os.path.basename(config.get('model', '?'))}")
    print(f"Gate    : {'enabled' if config.get('gate_enabled', True) else 'DISABLED'}")
    print(f"Chunks  : {total}  RTF: {float(summary.get('rtf', 0)):.4f}  "
          f"accept_rate: {float(summary.get('accept_rate', 0)):.4f}")

    # Scene distribution
    print(f"\nScene distribution:")
    for s in SCENES:
        grp = by_scene.get(s, [])
        n = len(grp)
        if n == 0:
            continue
        bar = "#" * min(40, int(40 * n / total))
        print(f"  {s:<24} {n:4d}  ({100*n/total:5.1f}%)  {bar}")

    # Cross-tabulation: scene x gate decision
    print(f"\nScene x Gate decision (counts):")
    hdr = f"  {'Scene':<24}"
    for d in DECISIONS:
        hdr += f"  {d:>10}"
    hdr += f"  {'no_asr':>8}  {'total':>6}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for s in SCENES:
        grp = by_scene.get(s, [])
        if not grp:
            continue
        row = f"  {s:<24}"
        for d in DECISIONS:
            row += f"  {cross[s].get(d, 0):>10}"
        no_asr = sum(1 for c in grp if not c.get("transcript") and float(c.get("infer_ms", 0)) == 0)
        row += f"  {no_asr:>8}  {len(grp):>6}"
        print(row)

    # Per-scene feature stats (rms, flatness, centroid, band_mid)
    print(f"\nPer-scene feature summary (mean | p10 | p90):")
    feat_keys = [("rms", ".4f"), ("flatness", ".4f"),
                 ("centroid_hz", ".0f"), ("band_mid", ".3f"), ("active_frac", ".3f")]
    for s in SCENES:
        grp = by_scene.get(s, [])
        if not grp:
            continue
        print(f"  {s}")
        for fk, fmt in feat_keys:
            vals = [c[fk] for c in grp if fk in c]
            if not vals:
                continue
            import statistics
            m_val = statistics.mean(vals)
            lo = pct(vals, 0.10)
            hi = pct(vals, 0.90)
            print(f"    {fk:<14}  mean={m_val:{fmt}}  p10={lo:{fmt}}  p90={hi:{fmt}}")

    # WER per scene
    if ref_map:
        print(f"\nWER per scene:")
        for s in SCENES:
            grp = by_scene.get(s, [])
            if not grp:
                continue
            w = _wer_for_chunks(grp, ref_map)
            wv = f"{w:.4f}" if w is not None else "n/a (no ref coverage)"
            print(f"  {s:<24}  WER={wv}")

    # ASR suppression summary (adaptive controller effect)
    n_no_asr = sum(
        1 for c in chunks
        if (c.get("scene") in ("MUSIC", "SILENCE", "NOISE")) and
           float(c.get("infer_ms", 0)) == 0
    )
    n_non_speech = sum(len(by_scene.get(s, [])) for s in ("MUSIC", "SILENCE", "NOISE"))
    if n_non_speech > 0:
        print(f"\nAdaptive ASR suppression: {n_no_asr}/{n_non_speech} "
              f"({100*n_no_asr/n_non_speech:.1f}%) non-speech chunks skipped ASR")


def _csv_output(path, chunks, by_scene, cross, ref_map, config, summary):
    """Emit one row per scene label."""
    import statistics
    fname = os.path.basename(path)
    header_printed = not hasattr(_csv_output, "_header_done")
    if not hasattr(_csv_output, "_header_done"):
        print("file,scene,count,pct_total,pass_n,border_n,fail_n,no_asr,"
              "rms_mean,flatness_mean,centroid_mean,band_mid_mean,wer")
        _csv_output._header_done = True

    total = len(chunks)
    for s in SCENES:
        grp = by_scene.get(s, [])
        if not grp:
            continue
        n = len(grp)
        rms_m = statistics.mean([c["rms"] for c in grp if "rms" in c]) if grp else 0
        flat_m = statistics.mean([c["flatness"] for c in grp if "flatness" in c]) if grp else 0
        cen_m = statistics.mean([c["centroid_hz"] for c in grp if "centroid_hz" in c]) if grp else 0
        bm_m = statistics.mean([c["band_mid"] for c in grp if "band_mid" in c]) if grp else 0
        no_asr = sum(1 for c in grp if float(c.get("infer_ms", 0)) == 0)
        wer_v = _wer_for_chunks(grp, ref_map) if ref_map else None
        wer_s = f"{wer_v:.4f}" if wer_v is not None else ""
        print(f"{fname},{s},{n},{100*n/total:.1f},"
              f"{cross[s].get('PASS', 0)},{cross[s].get('BORDERLINE', 0)},{cross[s].get('FAIL', 0)},"
              f"{no_asr},{rms_m:.4f},{flat_m:.4f},{cen_m:.1f},{bm_m:.4f},{wer_s}")


def load_refs_simple(path: str) -> Dict[int, str]:
    refs = {}
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.rstrip("\n")
            if "\t" in line:
                idx_str, _, text = line.partition("\t")
                try:
                    refs[int(idx_str)] = text.strip()
                except ValueError:
                    pass
            else:
                refs[i] = line.strip()
    return refs


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", help="JSON bench file(s)")
    parser.add_argument("--ref", default="",
                        help="Reference transcript file for per-scene WER")
    parser.add_argument("--csv", action="store_true", help="Output CSV")
    args = parser.parse_args()

    ref_map: Dict[int, str] = {}
    if args.ref:
        if not os.path.isfile(args.ref):
            print(f"WARNING: ref file not found: {args.ref}", file=sys.stderr)
        else:
            ref_map = load_refs_simple(args.ref)

    for path in args.files:
        if not os.path.isfile(path):
            print(f"WARNING: not found: {path}", file=sys.stderr)
            continue
        try:
            analyze(path, ref_map, args.csv)
        except Exception as e:
            print(f"ERROR analyzing {path}: {e}", file=sys.stderr)
            import traceback; traceback.print_exc()

    if not args.csv:
        print()


if __name__ == "__main__":
    main()
