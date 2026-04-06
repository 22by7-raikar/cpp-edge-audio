#!/usr/bin/env python3
"""
rtf_plot.py
Thread-scaling and optimization variant analysis from bench JSON files.

Produces three views:

  1. RTF table: one row per run, sorted by model / build label / threads.
  2. Thread-scaling table: linear speedup and parallel efficiency
     (for runs sharing the same model + build label, varying thread count).
  3. Quantization comparison: RTF and accept_rate across model variants
     for the same input / thread count.

Usage:
    python tools/python/eval/rtf_plot.py benchmarks/results/m6/*.json
    python tools/python/eval/rtf_plot.py benchmarks/results/m6/*.json --sort rtf
    python tools/python/eval/rtf_plot.py benchmarks/results/m6/*.json --csv > out.csv
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Optional


def load_run(path: str) -> Optional[dict]:
    try:
        with open(path) as f:
            data = json.load(f)
        config  = data.get("config", {})
        summary = data.get("summary", {})
        fname   = os.path.basename(path)
        return {
            "file":         fname,
            "model":        os.path.basename(config.get("model", "?")),
            "input":        os.path.basename(config.get("input", "?")),
            "chunk_ms":     int(config.get("chunk_ms", 0)),
            "n_threads":    int(config.get("n_threads", 1)),
            "gate_enabled": bool(config.get("gate_enabled", True)),
            "total_chunks": int(summary.get("total_chunks", 0)),
            "passed":       int(summary.get("passed", 0)),
            "borderline":   int(summary.get("borderline", 0)),
            "failed":       int(summary.get("failed", 0)),
            "accept_rate":  float(summary.get("accept_rate", 0)),
            "audio_sec":    float(summary.get("audio_sec", 0)),
            "total_infer_ms": float(summary.get("total_infer_ms", 0)),
            "rtf":          float(summary.get("rtf", 0)),
        }
    except Exception as e:
        print(f"WARNING: could not load {path}: {e}", file=sys.stderr)
        return None


def infer_build_label(fname: str) -> str:
    """
    Heuristic: extract build label from bench filename.
    Expected pattern: <wav>__<model>__<label>__t<N>__...
    Labels injected by benchmark_m6.sh: 'std', 'opt', 'prof'.
    Falls back to 'unknown'.
    """
    m = re.search(r'__(std|opt|prof)__', fname)
    return m.group(1) if m else "unknown"


def infer_quant(model_name: str) -> str:
    """
    Extract quantization level from ggml model filename.
    Examples: ggml-base.en.bin -> 'f16', ggml-base.en-q4_0.bin -> 'q4_0'
    """
    m = re.search(r'(q\d+[_k]?\d*)', model_name)
    return m.group(1) if m else "f16"


def _hbar(val: float, max_val: float, width: int = 30) -> str:
    if max_val <= 0:
        return ""
    n = min(width, int(width * val / max_val))
    return "|" + "#" * n + " " * (width - n) + "|"


def print_rtf_table(runs: List[dict], sort_key: str):
    sort_keys_map = {
        "rtf":         lambda r: r["rtf"],
        "threads":     lambda r: r["n_threads"],
        "accept_rate": lambda r: -r["accept_rate"],
        "model":       lambda r: r["model"],
        "file":        lambda r: r["file"],
    }
    key_fn = sort_keys_map.get(sort_key, sort_keys_map["file"])
    runs = sorted(runs, key=key_fn)

    max_rtf = max((r["rtf"] for r in runs), default=1.0)

    hdr = (f"  {'Model':<28}  {'build':>5}  {'t':>2}  "
           f"{'chunk':>5}  {'RTF':>7}  {'accept':>7}  "
           f"{'asr_ms':>8}  {'chunks':>6}")
    print(f"\nRTF summary ({len(runs)} runs)")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for r in runs:
        bl = infer_build_label(r["file"])
        bar = _hbar(r["rtf"], max_rtf, 20)
        print(f"  {r['model']:<28}  {bl:>5}  {r['n_threads']:>2}  "
              f"{r['chunk_ms']:>5}  {r['rtf']:>7.4f}  {r['accept_rate']:>7.4f}  "
              f"{r['total_infer_ms']:>8.1f}  {r['total_chunks']:>6}  {bar}")


def print_scaling_table(runs: List[dict]):
    """
    For each (model, build_label, chunk_ms, gate) group, print thread-scaling.
    Assumes the single-thread run is the baseline for linear speedup.
    """
    groups: Dict[tuple, List[dict]] = defaultdict(list)
    for r in runs:
        bl = infer_build_label(r["file"])
        key = (r["model"], bl, r["chunk_ms"], r["gate_enabled"])
        groups[key].append(r)

    any_printed = False
    for key, grp in sorted(groups.items()):
        grp.sort(key=lambda r: r["n_threads"])
        if len(grp) < 2:
            continue  # need at least 2 thread counts to show scaling

        model, build, chunk_ms, gate = key
        # Baseline = lowest thread count in group
        base = grp[0]
        base_rtf = base["rtf"]
        if base_rtf <= 0:
            continue

        if not any_printed:
            print(f"\nThread-scaling efficiency")
        any_printed = True

        print(f"\n  Model={model}  build={build}  chunk={chunk_ms}ms  "
              f"gate={'on' if gate else 'off'}")
        print(f"  {'threads':>8}  {'RTF':>8}  {'speedup':>8}  {'efficiency':>10}  bar")
        for r in grp:
            speedup = base_rtf / r["rtf"] if r["rtf"] > 0 else 0.0
            # Ideal linear speedup = t / t_base
            t_ratio = r["n_threads"] / max(base["n_threads"], 1)
            efficiency = speedup / t_ratio if t_ratio > 0 else 0.0
            bar = _hbar(efficiency, 1.0, 20)
            print(f"  {r['n_threads']:>8}  {r['rtf']:>8.4f}  {speedup:>8.2f}x  "
                  f"{efficiency:>9.1%}  {bar}")

    if not any_printed:
        print("\nNo thread-scaling groups found (need same model at multiple thread counts).")


def print_quant_table(runs: List[dict]):
    """
    Group by (input, build_label, chunk_ms, n_threads) and compare models
    side by side to show quantization tradeoff.
    """
    groups: Dict[tuple, List[dict]] = defaultdict(list)
    for r in runs:
        bl = infer_build_label(r["file"])
        key = (r["input"], bl, r["chunk_ms"], r["n_threads"], r["gate_enabled"])
        groups[key].append(r)

    any_printed = False
    for key, grp in sorted(groups.items()):
        if len(grp) < 2:
            continue
        grp.sort(key=lambda r: r["rtf"])
        inp, build, chunk_ms, threads, gate = key

        if not any_printed:
            print(f"\nQuantization / model comparison")
        any_printed = True

        print(f"\n  input={inp}  build={build}  "
              f"chunk={chunk_ms}ms  t={threads}  gate={'on' if gate else 'off'}")
        print(f"  {'Model':<32}  {'quant':>6}  {'RTF':>8}  {'accept':>7}  {'infer_ms':>10}")

        fastest_rtf = grp[0]["rtf"]
        for r in grp:
            quant = infer_quant(r["model"])
            rel = f"({r['rtf'] / fastest_rtf:.2f}x)" if fastest_rtf > 0 else ""
            print(f"  {r['model']:<32}  {quant:>6}  {r['rtf']:>8.4f}  "
                  f"{r['accept_rate']:>7.4f}  {r['total_infer_ms']:>10.1f}  {rel}")

    if not any_printed:
        print("\nNo multi-model groups found for quant comparison.")


def csv_output(runs: List[dict]):
    print("file,model,quant,build,n_threads,chunk_ms,gate,rtf,"
          "accept_rate,total_infer_ms,total_chunks,audio_sec")
    for r in runs:
        bl = infer_build_label(r["file"])
        quant = infer_quant(r["model"])
        print(f"{r['file']},{r['model']},{quant},{bl},"
              f"{r['n_threads']},{r['chunk_ms']},"
              f"{'1' if r['gate_enabled'] else '0'},"
              f"{r['rtf']:.5f},{r['accept_rate']:.4f},"
              f"{r['total_infer_ms']:.1f},{r['total_chunks']},"
              f"{r['audio_sec']:.3f}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", help="JSON bench file(s)")
    parser.add_argument("--sort", default="file",
                        choices=["file", "rtf", "threads", "accept_rate", "model"],
                        help="Sort key for the RTF table (default: file)")
    parser.add_argument("--csv", action="store_true",
                        help="Emit CSV instead of formatted tables")
    args = parser.parse_args()

    runs = []
    for path in args.files:
        if not os.path.isfile(path):
            print(f"WARNING: not found: {path}", file=sys.stderr)
            continue
        r = load_run(path)
        if r:
            runs.append(r)

    if not runs:
        print("No valid runs loaded.", file=sys.stderr)
        sys.exit(1)

    if args.csv:
        csv_output(runs)
        return

    print_rtf_table(runs, args.sort)
    print_scaling_table(runs)
    print_quant_table(runs)
    print()


if __name__ == "__main__":
    main()
