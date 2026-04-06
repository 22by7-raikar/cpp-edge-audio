#!/usr/bin/env python3
"""
sweep_thresholds.py
Grid-sweep gate thresholds against a stored bench JSON and report tradeoffs.

Requires a bench JSON produced with --no-gate so all chunks are present with
full metrics and transcripts. The script simulates gate decisions at every
(rms_min, flatness_max) grid point and reports:
  - Simulated accept rate
  - Estimated RTF savings vs inference on all chunks
  - Optionally: WER on accepted+borderline set if --ref is provided

The "recommended" operating point is the grid cell with maximum RTF savings
that keeps simulated accept rate >= --min-accept (default 0.85) and, if refs
are given, WER degradation < --max-wer-delta (default 0.03 absolute).

Usage:
    python tools/python/tuning/sweep_thresholds.py benchmarks/results/run_no_gate.json
    python tools/python/tuning/sweep_thresholds.py run.json --ref data/refs.tsv
    python tools/python/tuning/sweep_thresholds.py run.json --csv > sweep.csv
"""

import argparse
import json
import os
import sys
import statistics
from typing import Dict, List, Optional, Tuple


# Grid points to sweep
RMS_MIN_GRID = [0.0005, 0.001, 0.002, 0.003, 0.005, 0.010]
FLATNESS_MAX_GRID = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
SILENCE_MAX_GRID = [0.95]   # keep fixed for simplicity; extend if needed


def load_bench(path: str) -> Tuple[List[Dict], Dict, Dict]:
    with open(path) as f:
        data = json.load(f)
    chunks = data.get("chunks", [])
    for c in chunks:
        if "centroid_hz" not in c and "centroid" in c:
            c["centroid_hz"] = c["centroid"]
        if "active_frac" not in c and "active_frame_frac" in c:
            c["active_frac"] = c["active_frame_frac"]
    return chunks, data.get("config", {}), data.get("summary", {})


def simulate_gate(chunk: Dict,
                  rms_min: float,
                  flatness_max: float,
                  silence_max: float = 0.95,
                  clip_max: float = 0.05,
                  rms_borderline: Optional[float] = None) -> str:
    """
    Apply threshold rules to stored chunk metrics and return simulated decision.
    Mirrors the logic in gate.cpp without needing the C++ binary.
    """
    if rms_borderline is None:
        rms_borderline = rms_min * 0.33

    rms = chunk.get("rms", 0.0)
    silence = chunk.get("silence_ratio", 0.0)
    clip = chunk.get("clipping_ratio", 0.0)
    flatness = chunk.get("flatness", 0.0)

    if rms < rms_borderline:
        return "FAIL"
    if silence > silence_max:
        return "FAIL"
    if clip > clip_max:
        return "FAIL"
    if rms < rms_min:
        return "BORDERLINE"
    if flatness > flatness_max:
        return "FAIL"
    if flatness > flatness_max * 0.833:
        return "BORDERLINE"
    return "PASS"


def _wer_on_accepted(chunks: List[Dict],
                     ref_map: Dict[int, str],
                     accepted_set: set) -> Optional[float]:
    """Compute WER for chunks in accepted_set that have reference coverage."""
    total_ref = 0
    total_err = 0
    for c in chunks:
        idx = c.get("idx", -1)
        if idx not in accepted_set:
            continue
        ref = ref_map.get(idx, "")
        if not ref:
            continue
        hyp = c.get("transcript", "") or ""
        words_ref = ref.lower().split()
        words_hyp = hyp.lower().split()
        # Simple DP WER
        n, m = len(words_ref), len(words_hyp)
        dp = list(range(n + 1))
        for j in range(1, m + 1):
            prev = dp[:]
            dp[0] = j
            for i in range(1, n + 1):
                if words_ref[i - 1] == words_hyp[j - 1]:
                    dp[i] = prev[i - 1]
                else:
                    dp[i] = 1 + min(prev[i - 1], prev[i], dp[i - 1])
        total_ref += n
        total_err += dp[n]
    if total_ref == 0:
        return None
    return total_err / total_ref


def sweep(chunks: List[Dict],
          config: Dict,
          summary: Dict,
          ref_map: Dict[int, str],
          csv_mode: bool,
          min_accept: float,
          max_wer_delta: float,
          silence_max: float):

    total = len(chunks)
    if total == 0:
        print("No chunks found.", file=sys.stderr)
        return

    actual_rtf = float(summary.get("rtf", 0.0))
    baseline_accept = float(summary.get("accept_rate", 1.0))

    # Baseline WER on all chunks (gate-off run, all accepted)
    all_accepted_set = {c.get("idx", i) for i, c in enumerate(chunks)}
    baseline_wer = _wer_on_accepted(chunks, ref_map, all_accepted_set)

    # Header
    cols = ["rms_min", "flatness_max",
            "accept_rate", "pass_pct", "border_pct", "fail_pct",
            "estim_rtf", "rtf_saved_pct"]
    if ref_map:
        cols += ["wer", "wer_delta"]
    cols += ["flag"]

    if csv_mode:
        print(",".join(cols))
    else:
        print(f"\n{'=' * 72}")
        print(f"File summary: {total} chunks  actual_rtf={actual_rtf:.4f}  "
              f"baseline_accept={baseline_accept:.4f}")
        if baseline_wer is not None:
            print(f"Baseline WER (all accepted): {baseline_wer:.4f}")
        print(f"min_accept={min_accept:.2f}  silence_max={silence_max:.2f}")
        if ref_map:
            print(f"max_wer_delta={max_wer_delta:.3f}")
        print()
        hdr = (f"{'rms_min':>8}  {'flat_max':>8}  "
               f"{'accept':>7}  {'PASS%':>6}  {'BORD%':>6}  {'FAIL%':>6}  "
               f"{'est_RTF':>8}  {'saved%':>7}")
        if ref_map:
            hdr += f"  {'WER':>6}  {'dWER':>6}"
        hdr += "  flag"
        print(hdr)
        print("-" * len(hdr))

    best_row = None
    best_savings = -1.0

    for rms_min in RMS_MIN_GRID:
        for flatness_max in FLATNESS_MAX_GRID:
            decisions = [
                simulate_gate(c, rms_min, flatness_max, silence_max=silence_max)
                for c in chunks
            ]

            n_pass = decisions.count("PASS")
            n_border = decisions.count("BORDERLINE")
            n_fail = decisions.count("FAIL")
            n_accepted = n_pass + n_border
            accept_rate = n_accepted / total

            # Chunks that run ASR in simulation = PASS + BORDERLINE
            # RTF scales linearly with fraction of chunks sent to ASR
            frac_infer = n_accepted / total
            estim_rtf = actual_rtf * (frac_infer / max(baseline_accept, 1e-9))
            saved_pct = max(0.0, (1.0 - frac_infer / max(baseline_accept, 1e-9))) * 100.0

            wer_val: Optional[float] = None
            wer_delta: Optional[float] = None
            if ref_map:
                accepted_set = {
                    c.get("idx", i) for i, c in enumerate(chunks)
                    if decisions[i] in ("PASS", "BORDERLINE")
                }
                wer_val = _wer_on_accepted(chunks, ref_map, accepted_set)
                if wer_val is not None and baseline_wer is not None:
                    wer_delta = wer_val - baseline_wer

            # Determine flag
            flag = ""
            if accept_rate < min_accept:
                flag = "low_accept"
            elif ref_map and wer_delta is not None and wer_delta > max_wer_delta:
                flag = "wer_degraded"
            else:
                flag = "ok"
                if saved_pct > best_savings:
                    best_savings = saved_pct
                    best_row = (rms_min, flatness_max, accept_rate,
                                n_pass / total, n_border / total, n_fail / total,
                                estim_rtf, saved_pct, wer_val, wer_delta)

            if csv_mode:
                row = [f"{rms_min:.4f}", f"{flatness_max:.4f}",
                       f"{accept_rate:.4f}",
                       f"{n_pass / total:.4f}",
                       f"{n_border / total:.4f}",
                       f"{n_fail / total:.4f}",
                       f"{estim_rtf:.4f}",
                       f"{saved_pct:.2f}"]
                if ref_map:
                    row += [f"{wer_val:.4f}" if wer_val is not None else "",
                            f"{wer_delta:+.4f}" if wer_delta is not None else ""]
                row.append(flag)
                print(",".join(row))
            else:
                wer_str = ""
                if ref_map:
                    wv = f"{wer_val:.4f}" if wer_val is not None else "  n/a"
                    dw = f"{wer_delta:+.4f}" if wer_delta is not None else "  n/a"
                    wer_str = f"  {wv:>6}  {dw:>6}"
                marker = " <-- RECOMMENDED" if flag == "ok" and saved_pct == best_savings else ""
                print(f"{rms_min:>8.4f}  {flatness_max:>8.4f}  "
                      f"{accept_rate:>7.4f}  {n_pass / total:>6.2%}  "
                      f"{n_border / total:>6.2%}  {n_fail / total:>6.2%}  "
                      f"{estim_rtf:>8.4f}  {saved_pct:>6.1f}%"
                      f"{wer_str}  {flag}{marker}")

    if not csv_mode and best_row is not None:
        r, f, ar, pp, bp, fp, er, sp, wv, wd = best_row
        print(f"\nRecommended operating point:")
        print(f"  rms_min={r:.4f}  flatness_max={f:.4f}")
        print(f"  accept_rate={ar:.4f}  PASS={pp:.2%}  BORDERLINE={bp:.2%}  FAIL={fp:.2%}")
        print(f"  estimated_rtf={er:.4f}  inference_saved={sp:.1f}%")
        if wv is not None:
            print(f"  WER={wv:.4f}  delta={wd:+.4f}")
        print(f"\nTo apply, run with:")
        print(f"  ./audio_pipeline --rms-min {r} --max-flatness {f} ...")


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
    parser.add_argument("file", help="JSON bench file (preferably produced with --no-gate)")
    parser.add_argument("--ref", default="",
                        help="Reference transcript file (TSV or plain-text) for WER tracking")
    parser.add_argument("--csv", action="store_true", help="Output CSV instead of table")
    parser.add_argument("--min-accept", type=float, default=0.85,
                        help="Minimum acceptable accept rate (default: 0.85)")
    parser.add_argument("--max-wer-delta", type=float, default=0.03,
                        help="Maximum tolerated absolute WER increase (default: 0.03)")
    parser.add_argument("--silence-max", type=float, default=0.95,
                        help="Silence ratio threshold held fixed during sweep (default: 0.95)")
    args = parser.parse_args()

    if not os.path.isfile(args.file):
        print(f"ERROR: file not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    chunks, config, summary = load_bench(args.file)

    ref_map: Dict[int, str] = {}
    if args.ref:
        if not os.path.isfile(args.ref):
            print(f"WARNING: ref file not found: {args.ref}", file=sys.stderr)
        else:
            ref_map = load_refs_simple(args.ref)

    sweep(chunks, config, summary, ref_map,
          csv_mode=args.csv,
          min_accept=args.min_accept,
          max_wer_delta=args.max_wer_delta,
          silence_max=args.silence_max)


if __name__ == "__main__":
    main()
