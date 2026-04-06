#!/usr/bin/env python3
"""
wer.py
Compute Word Error Rate between reference transcripts and pipeline output.

Reference file format (--ref): one line per chunk, tab-separated:
    <chunk_idx>\t<reference text>
Or plain text (one line per chunk index 0, 1, 2, ... in order).

Hypothesis source (--hyp): JSON bench file produced by --bench-json.
Chunks with no transcript (gate rejected) count as empty hypothesis.

Usage:
    python tools/python/eval/wer.py --ref data/refs.tsv --hyp benchmarks/results/run.json
    python tools/python/eval/wer.py --ref data/refs.tsv --hyp a.json b.json --verbose
    python tools/python/eval/wer.py --ref data/refs.tsv --hyp a.json --skip-empty
"""

import argparse
import json
import os
import sys
import re
from typing import Dict, List, Tuple, Optional


# ---------------------------------------------------------------------------
# WER implementation (no external dependency)
# Uses dynamic programming edit distance on word sequences.
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _edit_distance(ref: List[str], hyp: List[str]) -> Tuple[int, int, int, int]:
    """
    Returns (substitutions, deletions, insertions, ref_len).
    Standard WER edit distance (Levenshtein on word sequences).
    """
    n, m = len(ref), len(hyp)
    # dp[i][j] = edit cost to align ref[:i] with hyp[:j]
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): dp[i][0] = i
    for j in range(m + 1): dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j - 1],  # substitution
                                    dp[i - 1][j],       # deletion
                                    dp[i][j - 1])       # insertion

    # Traceback to count operation types
    i, j = n, m
    subs = dels = ins = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i-1] == hyp[j-1]:
            i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            subs += 1; i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            dels += 1; i -= 1
        else:
            ins += 1; j -= 1

    return subs, dels, ins, n


def wer_score(ref: str, hyp: str) -> Dict:
    ref_words = _normalize(ref).split()
    hyp_words = _normalize(hyp).split()

    if not ref_words:
        return {"wer": 0.0, "subs": 0, "dels": 0, "ins": 0,
                "ref_len": 0, "hyp_len": len(hyp_words), "skipped": True}

    subs, dels, ins, ref_len = _edit_distance(ref_words, hyp_words)
    errors = subs + dels + ins
    wer = errors / ref_len if ref_len > 0 else 0.0

    return {
        "wer":     wer,
        "subs":    subs,
        "dels":    dels,
        "ins":     ins,
        "ref_len": ref_len,
        "hyp_len": len(hyp_words),
        "skipped": False,
    }


# ---------------------------------------------------------------------------
# Reference file loader
# ---------------------------------------------------------------------------

def load_refs(path: str) -> Dict[int, str]:
    """
    Load reference transcripts. Returns {chunk_idx: text}.
    Supports two formats:
      - Tab-separated: <idx>\t<text>
      - Plain text:    line 0 = chunk 0, line 1 = chunk 1, ...
    """
    refs = {}
    with open(path) as f:
        lines = [l.rstrip("\n") for l in f]

    # Detect tab-separated format
    if lines and "\t" in lines[0]:
        for line in lines:
            if not line.strip():
                continue
            idx_str, _, text = line.partition("\t")
            try:
                refs[int(idx_str.strip())] = text.strip()
            except ValueError:
                pass
    else:
        for i, line in enumerate(lines):
            refs[i] = line.strip()

    return refs


# ---------------------------------------------------------------------------
# JSON bench loader
# ---------------------------------------------------------------------------

def load_hyp(path: str) -> Tuple[Dict[int, str], Dict]:
    """Returns ({chunk_idx: transcript}, summary_dict)."""
    with open(path) as f:
        data = json.load(f)

    hyps = {}
    for c in data.get("chunks", []):
        idx  = c.get("idx", -1)
        text = c.get("transcript", "") or ""
        hyps[idx] = text

    return hyps, data.get("summary", {}), data.get("config", {})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def score_file(ref_path: str, hyp_path: str,
               verbose: bool, skip_empty: bool) -> Optional[Dict]:
    refs = load_refs(ref_path)
    hyps, summary, config = load_hyp(hyp_path)

    if not refs:
        print(f"  WARNING: no references loaded from {ref_path}", file=sys.stderr)
        return None

    total_subs = total_dels = total_ins = total_ref = 0
    n_skipped = n_empty_hyp = n_scored = 0
    per_chunk = []

    all_keys = sorted(set(refs) | set(hyps))

    for idx in all_keys:
        ref_text = refs.get(idx, "")
        hyp_text = hyps.get(idx, "")  # empty string if gate rejected

        if not ref_text:
            continue

        if skip_empty and not hyp_text:
            n_skipped += 1
            continue

        if not hyp_text:
            n_empty_hyp += 1

        r = wer_score(ref_text, hyp_text)
        total_subs += r["subs"]
        total_dels += r["dels"]
        total_ins  += r["ins"]
        total_ref  += r["ref_len"]
        n_scored   += 1

        per_chunk.append({
            "idx":      idx,
            "ref":      ref_text,
            "hyp":      hyp_text,
            "wer":      r["wer"],
            "subs":     r["subs"],
            "dels":     r["dels"],
            "ins":      r["ins"],
            "ref_len":  r["ref_len"],
        })

        if verbose:
            status = "EMPTY" if not hyp_text else f"WER={r['wer']:.3f}"
            print(f"  [{idx:3d}] {status}")
            print(f"        REF: {ref_text[:80]}")
            print(f"        HYP: {hyp_text[:80]}")

    overall_wer = (total_subs + total_dels + total_ins) / total_ref \
                  if total_ref > 0 else 0.0

    return {
        "file":          os.path.basename(hyp_path),
        "model":         os.path.basename(config.get("model", "?")),
        "threads":       config.get("n_threads", "?"),
        "chunk_ms":      config.get("chunk_ms", "?"),
        "gate":          "on" if config.get("gate_enabled", True) else "off",
        "n_refs":        len(refs),
        "n_scored":      n_scored,
        "n_skipped":     n_skipped,
        "n_empty_hyp":   n_empty_hyp,
        "total_ref_words": total_ref,
        "subs":          total_subs,
        "dels":          total_dels,
        "ins":           total_ins,
        "wer":           overall_wer,
        "rtf":           float(summary.get("rtf", 0.0)),
        "accept_rate":   float(summary.get("accept_rate", 0.0)),
        "per_chunk":     per_chunk,
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ref",  required=True,
                        help="Reference transcript file")
    parser.add_argument("--hyp",  required=True, nargs="+",
                        help="JSON bench file(s) from --bench-json")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print per-chunk ref/hyp comparison")
    parser.add_argument("--skip-empty", action="store_true",
                        help="Exclude gate-rejected (empty) chunks from WER")
    parser.add_argument("--csv", action="store_true",
                        help="Emit CSV summary table")
    args = parser.parse_args()

    if not os.path.isfile(args.ref):
        print(f"ERROR: ref file not found: {args.ref}", file=sys.stderr)
        sys.exit(1)

    results = []
    for hyp_path in args.hyp:
        if not os.path.isfile(hyp_path):
            print(f"WARNING: not found: {hyp_path}", file=sys.stderr)
            continue
        print(f"\n--- {os.path.basename(hyp_path)} ---")
        r = score_file(args.ref, hyp_path, args.verbose, args.skip_empty)
        if r:
            results.append(r)

    if not results:
        sys.exit(1)

    if args.csv:
        keys = ["file", "model", "threads", "chunk_ms", "gate",
                "n_scored", "n_empty_hyp", "wer", "rtf", "accept_rate",
                "subs", "dels", "ins", "total_ref_words"]
        print("\n" + ",".join(keys))
        for r in results:
            print(",".join(str(r[k]) for k in keys))
    else:
        print("\n" + "=" * 72)
        print(f"  {'file':<40} {'gate':4} {'WER':>7}  {'RTF':>7}  {'accept':>7}  {'scored':>6}")
        print("  " + "-" * 70)
        for r in results:
            fname = r["file"][:40]
            print(f"  {fname:<40} {r['gate']:4} {r['wer']:7.4f}  "
                  f"{r['rtf']:7.4f}  {r['accept_rate']:7.4f}  {r['n_scored']:6d}")
        print()

        if len(results) > 1:
            import statistics
            wers = [r["wer"] for r in results]
            print(f"  WER mean={statistics.mean(wers):.4f}  "
                  f"min={min(wers):.4f}  max={max(wers):.4f}")


if __name__ == "__main__":
    main()
