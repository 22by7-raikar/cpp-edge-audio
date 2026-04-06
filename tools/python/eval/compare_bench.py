#!/usr/bin/env python3
"""
compare_bench.py
Load benchmark JSON files produced by audio_pipeline --bench-json and
print a comparison table.

Usage:
    python tools/python/eval/compare_bench.py benchmarks/results/*.json
    python tools/python/eval/compare_bench.py benchmarks/results/*.json --sort rtf
    python tools/python/eval/compare_bench.py benchmarks/results/*.json --csv > results.csv

Sort key options (default: rtf):
    rtf, accept_rate, threads, chunk_ms, model, gate, infer_ms

Output columns:
    run_label  model  threads  chunk_ms  gate  rtf  accept_rate
    passed  failed  borderline  audio_sec  infer_ms  mean_flatness  mean_rms
"""

import json
import sys
import os
import argparse
import statistics
from typing import Dict, List, Optional


def load_bench(path: str) -> Optional[dict]:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"WARNING: could not load {path}: {e}", file=sys.stderr)
        return None


def parse_bench(path: str, data: dict) -> dict:
    cfg     = data.get("config", {})
    summary = data.get("summary", {})
    chunks  = data.get("chunks", [])

    # Aggregate per-chunk metrics for comparison
    flatness_vals = [c.get("flatness", 0.0)     for c in chunks]
    rms_vals      = [c.get("rms", 0.0)          for c in chunks]
    centroid_vals = [c.get("centroid_hz", 0.0)  for c in chunks]
    infer_vals    = [c.get("infer_ms", 0.0)     for c in chunks if c.get("infer_ms", 0.0) > 0]

    def _mean(v):
        return statistics.mean(v) if v else 0.0

    model_basename = os.path.basename(cfg.get("model", "?"))
    # Strip common prefix "ggml-"
    if model_basename.startswith("ggml-"):
        model_basename = model_basename[5:]
    if model_basename.endswith(".bin"):
        model_basename = model_basename[:-4]

    return {
        "label":        os.path.basename(path).replace(".json", ""),
        "model":        model_basename,
        "threads":      cfg.get("n_threads", 0),
        "chunk_ms":     cfg.get("chunk_ms", 0),
        "gate":         "on" if cfg.get("gate_enabled", True) else "off",
        "rtf":          float(summary.get("rtf", 0.0)),
        "accept_rate":  float(summary.get("accept_rate", 0.0)),
        "total_chunks": int(summary.get("total_chunks", 0)),
        "passed":       int(summary.get("passed", 0)),
        "failed":       int(summary.get("failed", 0)),
        "borderline":   int(summary.get("borderline", 0)),
        "audio_sec":    float(summary.get("audio_sec", 0.0)),
        "infer_ms":     float(summary.get("total_infer_ms", 0.0)),
        "mean_flatness": _mean(flatness_vals),
        "mean_rms":      _mean(rms_vals),
        "mean_centroid": _mean(centroid_vals),
        "mean_chunk_infer_ms": _mean(infer_vals),
    }


def print_table(rows: List[Dict], sort_key: str):
    if not rows:
        print("No results to display.")
        return

    rows.sort(key=lambda r: (r["model"], r["gate"], r["chunk_ms"], r[sort_key]))

    cols = [
        ("model",       "model",       16),
        ("threads",     "thr",          4),
        ("chunk_ms",    "chunk",        6),
        ("gate",        "gate",         4),
        ("rtf",         "RTF",          7),
        ("accept_rate", "accept",       7),
        ("passed",      "pass",         5),
        ("failed",      "fail",         5),
        ("borderline",  "bord",         5),
        ("audio_sec",   "audio_s",      8),
        ("infer_ms",    "infer_ms",    10),
        ("mean_flatness","flatness",     9),
        ("mean_rms",    "rms",          8),
    ]

    header = "  ".join(name.ljust(w) for _, name, w in cols)
    sep    = "  ".join("-" * w       for _, _,    w in cols)
    print(header)
    print(sep)

    for r in rows:
        parts = []
        for key, _, w in cols:
            val = r[key]
            if isinstance(val, float):
                s = f"{val:.4f}"
            else:
                s = str(val)
            parts.append(s.ljust(w))
        print("  ".join(parts))

    print()
    print(f"  {len(rows)} run(s) shown, sorted by {sort_key}")


def print_csv(rows: List[Dict]):
    if not rows:
        return
    keys = list(rows[0].keys())
    print(",".join(keys))
    for r in rows:
        print(",".join(str(r[k]) for k in keys))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", help="JSON bench files")
    parser.add_argument("--sort", default="rtf",
                        choices=["rtf", "accept_rate", "threads", "chunk_ms",
                                 "model", "gate", "infer_ms"],
                        help="Sort column (default: rtf)")
    parser.add_argument("--csv", action="store_true",
                        help="Emit CSV instead of a human-readable table")
    args = parser.parse_args()

    rows = []
    for path in args.files:
        if not os.path.isfile(path):
            print(f"WARNING: not found: {path}", file=sys.stderr)
            continue
        data = load_bench(path)
        if data:
            rows.append(parse_bench(path, data))

    if not rows:
        print("No valid results loaded.", file=sys.stderr)
        sys.exit(1)

    if args.csv:
        print_csv(rows)
    else:
        print(f"\nBenchmark comparison ({len(rows)} run(s))\n")
        print_table(rows, args.sort)

    # --- quick uplift summary: gate on vs off RTF delta ---
    gate_on   = [r for r in rows if r["gate"] == "on"]
    gate_off  = [r for r in rows if r["gate"] == "off"]
    if gate_on and gate_off:
        avg_on  = statistics.mean(r["rtf"] for r in gate_on)
        avg_off = statistics.mean(r["rtf"] for r in gate_off)
        print(f"  Gate-on  mean RTF : {avg_on:.4f}")
        print(f"  Gate-off mean RTF : {avg_off:.4f}")
        delta_pct = 100.0 * (avg_off - avg_on) / (avg_off + 1e-9)
        direction = "faster" if avg_on < avg_off else "slower"
        print(f"  Gate saves         {abs(delta_pct):.1f}% inference time ({direction})")

    print()


if __name__ == "__main__":
    main()
