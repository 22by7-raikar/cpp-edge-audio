#!/usr/bin/env python3
"""
train_quality_model.py
Lightweight learned quality predictor for gate transcription worthiness.

Features  : per-file mean+max of 12 gate metrics + gate decision fractions = 27.
Models    : LogisticRegression, RandomForestClassifier, GradientBoostingClassifier.
Split     : grouped by base_utterance_id to prevent utterance leakage across split.
Baseline  : rule gate (PASS/BORDERLINE -> accept, FAIL -> reject).

Usage:
    python tools/python/training/train_quality_model.py
    python tools/python/training/train_quality_model.py \\
        --labels data/labels/eval_subset.jsonl \\
        --out    benchmarks/results/quality_model/
"""

import argparse
import datetime
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "eval"))
import gate_eval  # noqa: E402

try:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
except ImportError:
    print(
        "ERROR: scikit-learn not installed.\n"
        "  conda install -n audio_king scikit-learn",
        file=sys.stderr,
    )
    sys.exit(1)

DEFAULT_LABELS = "data/labels/eval_subset.jsonl"
DEFAULT_OUT    = "benchmarks/results/quality_model/"
CHUNK_SEC      = 5.0
TRAIN_FRAC     = 0.6
SEED           = 42

# 12 gate metrics; must match keys returned by compute_chunk_metrics().
METRIC_KEYS: List[str] = [
    "rms", "silence_ratio", "clipping_ratio", "zcr",
    "flatness", "centroid_hz", "rolloff_hz", "flux",
    "band_low", "band_mid", "band_high", "active_frac",
]

# 27 features: 12 mean + 12 max + 3 gate-decision fractions.
FEATURE_NAMES: List[str] = (
    [f"{k}_mean" for k in METRIC_KEYS] +
    [f"{k}_max"  for k in METRIC_KEYS] +
    ["gate_pass_frac", "gate_borderline_frac", "gate_fail_frac"]
)


def _extract_features(path: str, cfg: Dict) -> Optional[Tuple[np.ndarray, str]]:
    """Run gate_eval on one file. Returns (27-d feature vector, file_decision) or None."""
    res = gate_eval.evaluate_file(path, cfg, CHUNK_SEC)
    if res.get("error") or not res.get("chunks"):
        return None

    chunks = res["chunks"]

    means = [
        float(np.mean([c[k] for c in chunks if k in c] or [0.0]))
        for k in METRIC_KEYS
    ]
    maxes = [
        float(np.max([c[k] for c in chunks if k in c] or [0.0]))
        for k in METRIC_KEYS
    ]

    decisions = [c.get("decision", "FAIL") for c in chunks]
    n = max(len(decisions), 1)
    fracs = [
        decisions.count("PASS")       / n,
        decisions.count("BORDERLINE") / n,
        decisions.count("FAIL")       / n,
    ]

    return np.array(means + maxes + fracs, dtype=np.float32), res["file_decision"]


def _make_split(
    entries: List[Dict],
    train_frac: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    """Return (train_indices, test_indices) grouped by base_utterance_id.

    Files derived from the same source utterance (speech_in_noise, reverb,
    clipped, etc.) share the same base_utterance_id. Grouping ensures those
    files all land in the same split, preventing content leakage.
    Files without a base_utterance_id (music, noise) use their path as key.
    """
    by_uid: Dict[str, List[int]] = {}
    for i, e in enumerate(entries):
        uid = e.get("base_utterance_id") or e["path"]
        by_uid.setdefault(uid, []).append(i)

    uids = sorted(by_uid.keys())
    np.random.default_rng(seed).shuffle(uids)
    n_tr = int(len(uids) * train_frac)

    train_idx = [i for uid in uids[:n_tr] for i in by_uid[uid]]
    test_idx  = [i for uid in uids[n_tr:] for i in by_uid[uid]]
    return train_idx, test_idx


def _eval_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    entries: List[Dict],
) -> Dict:
    """Compute FAR, FRR, precision, recall, F1, and per-label confusion counts."""
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / n_pos     if n_pos       else 0.0

    per_label: Dict[str, Dict] = {}
    for i, e in enumerate(entries):
        lbl = e["label"]
        d   = per_label.setdefault(lbl, {"tp": 0, "fp": 0, "tn": 0, "fn": 0})
        yt, yp = int(y_true[i]), int(y_pred[i])
        if   yt == 1 and yp == 1: d["tp"] += 1
        elif yt == 0 and yp == 1: d["fp"] += 1
        elif yt == 0 and yp == 0: d["tn"] += 1
        else:                     d["fn"] += 1

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "far":        round(fp / n_neg if n_neg else 0.0, 4),
        "frr":        round(fn / n_pos if n_pos else 0.0, 4),
        "precision":  round(prec, 4),
        "recall":     round(rec,  4),
        "f1":         round(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0, 4),
        "music_fa":        sum(
            1 for i, e in enumerate(entries)
            if e["label"] == "music" and y_pred[i] == 1
        ),
        "clean_speech_fr": sum(
            1 for i, e in enumerate(entries)
            if e["label"] == "clean_speech" and y_pred[i] == 0
        ),
        "per_label": per_label,
    }


def _print_report(results: List[Dict], n_tr: int, n_te: int) -> None:
    sep = "=" * 72
    print(f"\n{sep}")
    print("QUALITY MODEL EVALUATION  (sklearn prototype)")
    print(sep)
    print(f"Train: {n_tr}  Test: {n_te}  (split by base_utterance_id, seed={SEED})")
    print()

    hdr = (
        f"  {'Model':32s}  {'FAR':>6}  {'FRR':>6}  "
        f"{'P':>6}  {'R':>6}  {'F1':>6}  {'MusicFA':>8}  {'CleanFR':>8}"
    )
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in results:
        m = r["metrics"]
        print(
            f"  {r['name']:32s}  {m['far']:>6.4f}  {m['frr']:>6.4f}  "
            f"{m['precision']:>6.4f}  {m['recall']:>6.4f}  {m['f1']:>6.4f}  "
            f"{m['music_fa']:>8d}  {m['clean_speech_fr']:>8d}"
        )

    print()
    print("Per-label false decisions (test set)  [fp / fn]:")
    labels = sorted({lbl for r in results for lbl in r["metrics"]["per_label"]})
    col_w  = 12
    header = f"  {'Label':30s}"
    for r in results:
        header += f"  {r['name'][:col_w]:>{col_w}}"
    print(header)
    print("  " + "-" * (32 + (col_w + 2) * len(results)))
    for lbl in labels:
        row = f"  {lbl:30s}"
        for r in results:
            pl  = r["metrics"]["per_label"].get(lbl, {})
            row += f"  {pl.get('fp', 0):>4}/{pl.get('fn', 0):<4}  "
        print(row)

    # Feature importances from the first tree-based model that has them.
    for r in results:
        if "feature_importances" in r:
            print(f"\nTop-10 features ({r['name']}):")
            ranked = sorted(r["feature_importances"].items(), key=lambda x: -x[1])[:10]
            for fname, imp in ranked:
                print(f"    {fname:25s}  {imp:.4f}")
            break

    print()


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--labels",     default=DEFAULT_LABELS,
                    help=f"JSONL label file (default: {DEFAULT_LABELS})")
    ap.add_argument("--out",        default=DEFAULT_OUT,
                    help=f"Output directory (default: {DEFAULT_OUT})")
    ap.add_argument("--train-frac", type=float, default=TRAIN_FRAC,
                    help=f"Fraction of utterance groups in train (default: {TRAIN_FRAC})")
    ap.add_argument("--seed",       type=int,   default=SEED,
                    help=f"RNG seed (default: {SEED})")
    args = ap.parse_args()

    # Resolve relative paths from the repo root (3 levels up from this file).
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
    if not os.path.isabs(args.labels):
        args.labels = os.path.join(_repo, args.labels)

    print(f"Loading labels: {args.labels}", file=sys.stderr)
    records = [json.loads(l) for l in open(args.labels) if l.strip()]
    cfg = dict(gate_eval.DEFAULT_CONFIG)

    # -----------------------------------------------------------------------
    # Feature extraction
    # -----------------------------------------------------------------------
    print(f"Extracting features ({len(records)} files)...", file=sys.stderr)
    X_list:    List[np.ndarray] = []
    y_list:    List[int]        = []
    gate_list: List[int]        = []
    valid:     List[Dict]       = []

    for k, entry in enumerate(records):
        if k % 100 == 0:
            print(f"  {k}/{len(records)}", file=sys.stderr, flush=True)
        out = _extract_features(entry["path"], cfg)
        if out is None:
            print(f"  SKIP {os.path.basename(entry['path'])}", file=sys.stderr)
            continue
        feat, fdec = out
        X_list.append(feat)
        y_list.append(1 if entry["should_transcribe"] == "yes" else 0)
        gate_list.append(1 if fdec in ("PASS", "BORDERLINE") else 0)
        valid.append(entry)

    X          = np.array(X_list)      # (N, 27)
    y          = np.array(y_list)
    gate_preds = np.array(gate_list)
    print(f"  {len(valid)}/{len(records)} files extracted.", file=sys.stderr)

    # -----------------------------------------------------------------------
    # Train/test split
    # -----------------------------------------------------------------------
    tr_idx, te_idx = _make_split(valid, args.train_frac, args.seed)
    X_tr, X_te = X[tr_idx], X[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]
    gate_te    = gate_preds[te_idx]
    te_ents    = [valid[i] for i in te_idx]

    print(
        f"Train: {len(tr_idx)}"
        f"  (pos={int(y_tr.sum())} neg={len(y_tr) - int(y_tr.sum())})"
        f"  Test: {len(te_idx)}"
        f"  (pos={int(y_te.sum())} neg={len(y_te) - int(y_te.sum())})",
        file=sys.stderr,
    )

    # Scaled versions for logistic regression.
    scaler  = StandardScaler().fit(X_tr)
    X_tr_s  = scaler.transform(X_tr)
    X_te_s  = scaler.transform(X_te)

    # -----------------------------------------------------------------------
    # Train and evaluate
    # -----------------------------------------------------------------------
    all_results = [
        {"name": "rule_gate", "metrics": _eval_metrics(y_te, gate_te, te_ents)},
    ]

    model_specs = [
        (
            "logreg",
            LogisticRegression(C=1.0, solver="liblinear", max_iter=500, random_state=SEED),
            True,   # use scaled features
        ),
        (
            "rf",
            RandomForestClassifier(
                n_estimators=100, max_depth=6, min_samples_leaf=3,
                n_jobs=-1, random_state=SEED,
            ),
            False,
        ),
        (
            "gbt",
            GradientBoostingClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.1,
                random_state=SEED,
            ),
            False,
        ),
    ]

    for name, clf, use_scaled in model_specs:
        print(f"Training {name}...", file=sys.stderr)
        Xtr = X_tr_s if use_scaled else X_tr
        Xte = X_te_s if use_scaled else X_te
        clf.fit(Xtr, y_tr)
        yp  = clf.predict(Xte)
        mr  = {"name": name, "metrics": _eval_metrics(y_te, yp, te_ents)}
        if hasattr(clf, "feature_importances_"):
            mr["feature_importances"] = {
                n: round(float(v), 6)
                for n, v in zip(FEATURE_NAMES, clf.feature_importances_)
            }
        all_results.append(mr)

    _print_report(all_results, len(tr_idx), len(te_idx))

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    if args.out:
        out_dir = args.out if os.path.isabs(args.out) else os.path.join(_repo, args.out)
        os.makedirs(out_dir, exist_ok=True)
        ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(out_dir, f"quality_model_{ts}.json")
        with open(out_path, "w") as fh:
            json.dump(
                {
                    "ts":           ts,
                    "n_records":    len(records),
                    "n_valid":      len(valid),
                    "train_n":      len(tr_idx),
                    "test_n":       len(te_idx),
                    "train_frac":   args.train_frac,
                    "seed":         args.seed,
                    "chunk_sec":    CHUNK_SEC,
                    "feature_names": FEATURE_NAMES,
                    "models":       all_results,
                },
                fh,
                indent=2,
            )
        print(f"Results saved: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
