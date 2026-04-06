#!/usr/bin/env python3
"""
parse_logs.py
Parse structured pipeline log files (tab-separated key=value per line)
and print summary statistics.

Usage:
    python tools/python/eval/parse_logs.py benchmarks/results/sample_gate_on_5s_t4.log
    python tools/python/eval/parse_logs.py benchmarks/results/*.log
"""

import sys
import os
from collections import defaultdict


def parse_record(line: str) -> dict:
    parts = line.strip().split("\t")
    rec = {}
    for p in parts:
        if "=" in p:
            k, _, v = p.partition("=")
            rec[k] = v
    return rec


def load_log(path: str):
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(parse_record(line))
    return records


def summarize(path: str):
    records = load_log(path)

    run_start = next((r for r in records if r.get("event") == "run_start"), {})
    run_end   = next((r for r in records if r.get("event") == "run_end"),   {})
    chunks    = [r for r in records if r.get("event") == "chunk"]

    print(f"\n=== {os.path.basename(path)} ===")
    if run_start:
        print(f"  input      : {run_start.get('input', '?')}")
        print(f"  model      : {run_start.get('model', '?')}")
        print(f"  chunk_ms   : {run_start.get('chunk_ms', '?')}")
        print(f"  threads    : {run_start.get('threads', '?')}")
        print(f"  gate       : {'enabled' if run_start.get('gate') == '1' else 'disabled'}")

    if run_end:
        print(f"  total_chunks: {run_end.get('total_chunks', '?')}")
        print(f"  passed      : {run_end.get('passed', '?')}")
        print(f"  borderline  : {run_end.get('borderline', '?')}")
        print(f"  failed      : {run_end.get('failed', '?')}")
        print(f"  accept_rate : {run_end.get('accept_rate', '?')}")
        print(f"  audio_sec   : {run_end.get('audio_sec', '?')}")
        print(f"  infer_ms    : {run_end.get('infer_ms', '?')}")
        print(f"  rtf         : {run_end.get('rtf', '?')}")

    if not chunks:
        return

    # Per-decision breakdown
    decisions     = defaultdict(int)
    rms_vals      = []
    flatness_vals = []
    centroid_vals = []
    rolloff_vals  = []
    active_vals   = []

    for c in chunks:
        decisions[c.get("decision", "UNKNOWN")] += 1
        for key, store in [
            ("rms",      rms_vals),
            ("flatness", flatness_vals),
            ("centroid", centroid_vals),
            ("rolloff",  rolloff_vals),
            ("active",   active_vals),
        ]:
            try:
                store.append(float(c[key]))
            except (KeyError, ValueError):
                pass

    print(f"\n  Gate decisions: {dict(decisions)}")

    import statistics

    def _stats(label, vals):
        if not vals:
            return
        print(f"  {label:<20} mean={statistics.mean(vals):.4f}  "
              f"median={statistics.median(vals):.4f}  "
              f"min={min(vals):.4f}  max={max(vals):.4f}")

    _stats("rms",               rms_vals)
    _stats("spectral_flatness", flatness_vals)
    _stats("centroid_hz",       centroid_vals)
    _stats("rolloff_hz",        rolloff_vals)
    _stats("active_frame_frac", active_vals)

    # Print transcripts
    texts = [(c.get("idx", "?"), c.get("text", "")) for c in chunks if c.get("text")]
    if texts:
        print(f"\n  Transcripts ({len(texts)} chunks with text):")
        for idx, txt in texts[:10]:
            print(f"    [{idx}] {txt[:120]}")
        if len(texts) > 10:
            print(f"    ... ({len(texts) - 10} more)")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    for path in sys.argv[1:]:
        if os.path.isfile(path):
            summarize(path)
        else:
            print(f"File not found: {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
