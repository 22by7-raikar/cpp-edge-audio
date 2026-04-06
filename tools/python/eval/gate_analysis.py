#!/usr/bin/env python3
"""
gate_analysis.py
Analyze gate decision distributions and feature statistics from bench JSON files.

Produces:
  - Feature statistics (mean, p10, p50, p90) broken down by gate decision
  - Decision count summary
  - Optionally: per-decision WER breakdown if --ref is given
  - Flag distribution: which feature is most often the rejection trigger

Use a gate-off run (--no-gate) to get metrics for all chunks regardless of
what the gate would have decided, then compare against a gate-on run to
understand precision/recall tradeoffs before committing to a threshold.

Usage:
    python tools/python/eval/gate_analysis.py benchmarks/results/run.json
    python tools/python/eval/gate_analysis.py gate_on.json gate_off.json
    python tools/python/eval/gate_analysis.py run.json --ref data/refs.tsv
"""

import argparse
import json
import os
import sys
import statistics
from typing import Dict, List


FEATURES = [
    ("rms",             "RMS energy"),
    ("flatness",        "Spectral flatness  [0=tonal, 1=noise]"),
    ("silence_ratio",   "Silence ratio      [frac silent samples]"),
    ("clipping_ratio",  "Clipping ratio     [frac clipped samples]"),
    ("zcr",             "Zero-crossing rate [crossings/sec]"),
    ("centroid_hz",     "Spectral centroid  [Hz]"),
    ("rolloff_hz",      "Spectral rolloff   [Hz, 85% energy]"),
    ("active_frac",     "Active frame frac  [frames above rms_thresh]"),
    ("band_low",        "Band energy low    [0-500 Hz, normalized]"),
    ("band_mid",        "Band energy mid    [500-4000 Hz, normalized]"),
    ("band_high",       "Band energy high   [4000+ Hz, normalized]"),
]

DECISIONS = ["PASS", "BORDERLINE", "FAIL"]


def load_bench(path: str) -> List[Dict]:
    with open(path) as f:
        data = json.load(f)
    chunks = data.get("chunks", [])
    # Normalize centroid key name (JSON uses centroid_hz)
    for c in chunks:
        if "centroid_hz" not in c and "centroid" in c:
            c["centroid_hz"] = c["centroid"]
        if "rolloff_hz" not in c and "rolloff" in c:
            c["rolloff_hz"] = c["rolloff"]
        if "active_frac" not in c and "active_frame_frac" in c:
            c["active_frac"] = c["active_frame_frac"]
    return chunks, data.get("config", {}), data.get("summary", {})


def pct(vals: List[float], p: float) -> float:
    if not vals:
        return 0.0
    sv = sorted(vals)
    idx = (len(sv) - 1) * p
    lo, hi = int(idx), min(int(idx) + 1, len(sv) - 1)
    return sv[lo] + (sv[hi] - sv[lo]) * (idx - lo)


def feature_stats(vals: List[float]) -> str:
    if not vals:
        return "    n=0"
    return (f"n={len(vals):4d}  "
            f"mean={statistics.mean(vals):8.4f}  "
            f"p10={pct(vals, 0.10):8.4f}  "
            f"p50={pct(vals, 0.50):8.4f}  "
            f"p90={pct(vals, 0.90):8.4f}  "
            f"max={max(vals):8.4f}")


def analyze(path: str, ref_map: Dict[int, str]):
    chunks, config, summary = load_bench(path)

    print(f"\n{'=' * 72}")
    print(f"File    : {os.path.basename(path)}")
    model = os.path.basename(config.get("model", "?"))
    print(f"Model   : {model}")
    print(f"Gate    : {'enabled' if config.get('gate_enabled', True) else 'DISABLED'}")
    print(f"chunk_ms: {config.get('chunk_ms', '?')}  threads: {config.get('n_threads', '?')}")
    print(f"Chunks  : {summary.get('total_chunks', len(chunks))}  "
          f"RTF: {float(summary.get('rtf', 0)):.4f}  "
          f"accept_rate: {float(summary.get('accept_rate', 0)):.4f}")

    # Group chunks by decision
    by_decision: Dict[str, List[Dict]] = {d: [] for d in DECISIONS}
    by_decision["UNKNOWN"] = []
    reason_counts: Dict[str, int] = {}

    for c in chunks:
        dec = c.get("decision", "UNKNOWN")
        by_decision.setdefault(dec, []).append(c)
        reason = c.get("reason", "")
        if reason:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    # Decision summary
    print(f"\nDecision breakdown:")
    for d in DECISIONS:
        n = len(by_decision.get(d, []))
        bar = "#" * min(40, int(40 * n / max(len(chunks), 1)))
        print(f"  {d:<12} {n:4d}  {bar}")

    # Rejection reasons
    if reason_counts:
        print(f"\nRejection reasons:")
        for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
            print(f"  {reason:<28} {count:4d}")

    # Feature statistics by decision
    print(f"\nFeature statistics by decision:")
    for feat_key, feat_label in FEATURES:
        print(f"\n  {feat_label}")
        for d in DECISIONS:
            grp = by_decision.get(d, [])
            vals = [c[feat_key] for c in grp if feat_key in c]
            print(f"    {d:<12} {feature_stats(vals)}")

    # WER by decision (if refs provided)
    if ref_map:
        _wer_by_decision(chunks, by_decision, ref_map)

    # Threshold recommendations
    _recommend_thresholds(by_decision, chunks)


def _wer_by_decision(chunks, by_decision, ref_map):
    from wer import wer_score  # relative import from same directory

    print(f"\nWER by gate decision (using provided references):")
    for d in DECISIONS:
        grp = by_decision.get(d, [])
        if not grp:
            continue
        total_ref = total_err = 0
        for c in grp:
            idx = c.get("idx", -1)
            ref = ref_map.get(idx, "")
            hyp = c.get("transcript", "") or ""
            if not ref:
                continue
            r = wer_score(ref, hyp)
            total_ref += r["ref_len"]
            total_err += r["subs"] + r["dels"] + r["ins"]
        if total_ref > 0:
            wer = total_err / total_ref
            print(f"  {d:<12}  WER={wer:.4f}  (ref_words={total_ref})")
        else:
            print(f"  {d:<12}  no reference coverage")


def _recommend_thresholds(by_decision, all_chunks):
    """
    Print conservative threshold suggestions based on feature separation
    between PASS and FAIL groups.
    """
    pass_chunks = by_decision.get("PASS", [])
    fail_chunks = by_decision.get("FAIL", [])

    if not pass_chunks or not fail_chunks:
        return

    print(f"\nThreshold guidance (conservative: minimize false rejections):")

    def suggest(key, higher_is_worse: bool):
        pv = [c[key] for c in pass_chunks if key in c]
        fv = [c[key] for c in fail_chunks if key in c]
        if not pv or not fv:
            return
        if higher_is_worse:
            # Suggest threshold just above the 90th percentile of PASS distribution
            p_p90 = pct(pv, 0.90)
            f_p10 = pct(fv, 0.10)
            suggested = (p_p90 + f_p10) / 2.0
            overlap = sum(1 for v in pv if v > f_p10) / len(pv)
            print(f"  {key:<20}  PASS-p90={p_p90:.4f}  FAIL-p10={f_p10:.4f}  "
                  f"suggested_max={suggested:.4f}  pass_overlap={overlap:.2%}")
        else:
            # Suggest threshold just below the 10th percentile of PASS
            p_p10 = pct(pv, 0.10)
            f_p90 = pct(fv, 0.90)
            suggested = (p_p10 + f_p90) / 2.0
            overlap = sum(1 for v in pv if v < f_p90) / len(pv)
            print(f"  {key:<20}  PASS-p10={p_p10:.4f}  FAIL-p90={f_p90:.4f}  "
                  f"suggested_min={suggested:.4f}  pass_overlap={overlap:.2%}")

    suggest("rms",           higher_is_worse=False)
    suggest("flatness",      higher_is_worse=True)
    suggest("silence_ratio", higher_is_worse=True)
    suggest("active_frac",   higher_is_worse=False)


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
                        help="Reference transcript file for WER-by-decision breakdown")
    args = parser.parse_args()

    ref_map = {}
    if args.ref:
        if not os.path.isfile(args.ref):
            print(f"WARNING: ref file not found: {args.ref}", file=sys.stderr)
        else:
            ref_map = load_refs_simple(args.ref)

    for path in args.files:
        if not os.path.isfile(path):
            print(f"WARNING: not found: {path}", file=sys.stderr)
            continue
        # If --ref given and wer import available, use it
        try:
            analyze(path, ref_map)
        except Exception as e:
            print(f"ERROR analyzing {path}: {e}", file=sys.stderr)
            import traceback; traceback.print_exc()

    print()


if __name__ == "__main__":
    main()
