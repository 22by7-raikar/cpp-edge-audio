#!/usr/bin/env python3
"""
train_quality_model.py
Lightweight learned quality predictor for gate transcription worthiness.

Features  : per-file mean+max of 12 gate metrics + gate decision fractions = 27.
Models    : LogisticRegression, RandomForestClassifier, GradientBoostingClassifier.
Split     : grouped by base_utterance_id to prevent utterance leakage across split.
Baseline  : rule gate (PASS/BORDERLINE -> accept, FAIL -> reject).

Analyses:
    threshold_sweep   FAR/FRR/F1 at thresholds 0.1..0.9 (prob models only).
    ablation          DSP-only (24 features) vs all (27 features).
    operating_points  balanced (max F1) and conservative
                      (max F1 s.t. FRR <= rule_gate_FRR + 0.02).

Outputs written to --out:
    model_comparison_<ts>.json
    feature_importance_<ts>.json
    threshold_sweep_<ts>.tsv
    recommended_operating_points_<ts>.json

Usage:
    python tools/python/training/train_quality_model.py \\
        --train-labels data/labels/quality_train.jsonl \\
        --labels       data/labels/eval_subset.jsonl \\
        --out          benchmarks/results/quality_model/
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

# First 24 features are DSP-only (no gate-decision fractions).
N_DSP_FEATURES = 24

# Probability thresholds to sweep.
THRESHOLD_STEPS: List[float] = [round(t * 0.1, 1) for t in range(1, 10)]


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


def _extract_all(
    records: List[Dict],
    cfg: Dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
    """Extract features for all records.

    Returns (X, y, gate_preds, valid_entries).  Skips files that fail.
    gate_preds uses PASS/BORDERLINE -> 1, FAIL -> 0 (rule gate baseline).
    """
    X_list:    List[np.ndarray] = []
    y_list:    List[int]        = []
    gate_list: List[int]        = []
    valid:     List[Dict]       = []

    for k, entry in enumerate(records):
        if k % 100 == 0 and k > 0:
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

    n_feat = len(FEATURE_NAMES)
    if not X_list:
        return np.empty((0, n_feat)), np.array([]), np.array([]), []
    return np.array(X_list), np.array(y_list), np.array(gate_list), valid


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


def _sweep_thresholds(
    model_name: str,
    y_true: np.ndarray,
    y_proba: np.ndarray,
    entries: List[Dict],
    steps: List[float] = THRESHOLD_STEPS,
) -> List[Dict]:
    """Evaluate decision thresholds. y_proba is P(class=1)."""
    rows = []
    for t in steps:
        y_pred = (y_proba >= t).astype(int)
        m = _eval_metrics(y_true, y_pred, entries)
        rows.append({
            "model":           model_name,
            "threshold":       t,
            "far":             m["far"],
            "frr":             m["frr"],
            "precision":       m["precision"],
            "recall":          m["recall"],
            "f1":              m["f1"],
            "music_fa":        m["music_fa"],
            "clean_speech_fr": m["clean_speech_fr"],
        })
    return rows


def _recommend_operating_points(
    sweep_rows: List[Dict],
    rule_frr: float,
) -> Dict:
    """Return two operating points from a threshold sweep.

    balanced     : threshold maximising F1.
    conservative : threshold maximising F1 subject to FRR <= rule_frr + 0.02.
                   Falls back to minimum-FRR row when no row meets the constraint.
    """
    balanced = max(sweep_rows, key=lambda r: r["f1"])
    frr_limit = round(rule_frr + 0.02, 4)
    cands = [r for r in sweep_rows if r["frr"] <= frr_limit]
    conservative = max(cands, key=lambda r: r["f1"]) if cands else min(sweep_rows, key=lambda r: r["frr"])
    return {"balanced": balanced, "conservative": conservative}


def _print_report(results: List[Dict], n_tr: int, n_te: int, mode: str = "") -> None:
    sep = "=" * 72
    print(f"\n{sep}")
    print("QUALITY MODEL EVALUATION  (sklearn prototype)")
    print(sep)
    print(f"Train: {n_tr}  Test: {n_te}  {mode}")
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


def _print_ops_summary(ops_by_model: Dict[str, Dict], rule_metrics: Dict) -> None:
    """Print threshold-sweep operating point recommendations."""
    sep = "=" * 72
    print(f"\n{sep}")
    print("THRESHOLD SWEEP  -  RECOMMENDED OPERATING POINTS")
    print(sep)
    print(
        f"  Rule gate baseline:  FAR={rule_metrics['far']:.4f}  "
        f"FRR={rule_metrics['frr']:.4f}  F1={rule_metrics['f1']:.4f}"
    )
    print()
    hdr = (
        f"  {'Model':20s}  {'Point':13s}  {'Thresh':>6}  {'FAR':>6}  "
        f"{'FRR':>6}  {'F1':>6}  {'MusicFA':>8}  {'CleanFR':>8}"
    )
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for model_name, ops in ops_by_model.items():
        for point_name, row in ops.items():
            print(
                f"  {model_name:20s}  {point_name:13s}  {row['threshold']:>6.1f}  "
                f"{row['far']:>6.4f}  {row['frr']:>6.4f}  {row['f1']:>6.4f}  "
                f"{row['music_fa']:>8d}  {row['clean_speech_fr']:>8d}"
            )
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
    ap.add_argument("--train-labels", default=None,
                    help="Separate training JSONL; if set, --labels is eval-only")
    args = ap.parse_args()

    # Resolve relative paths from the repo root (3 levels up from this file).
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
    if not os.path.isabs(args.labels):
        args.labels = os.path.join(_repo, args.labels)
    if args.train_labels and not os.path.isabs(args.train_labels):
        args.train_labels = os.path.join(_repo, args.train_labels)

    print(f"Loading labels: {args.labels}", file=sys.stderr)
    records = [json.loads(l) for l in open(args.labels) if l.strip()]
    cfg = dict(gate_eval.DEFAULT_CONFIG)

    # -----------------------------------------------------------------------
    # Feature extraction
    # -----------------------------------------------------------------------
    if args.train_labels:
        # Cross-dataset mode: train on quality_train, eval on eval_subset.
        print(f"Loading train labels: {args.train_labels}", file=sys.stderr)
        train_recs = [json.loads(l) for l in open(args.train_labels) if l.strip()]
        print(f"Extracting train features ({len(train_recs)} files)...", file=sys.stderr)
        X_tr, y_tr, _, _tr_ents = _extract_all(train_recs, cfg)
        print(f"  {len(X_tr)}/{len(train_recs)} extracted.", file=sys.stderr)
        print(f"Extracting eval features ({len(records)} files)...", file=sys.stderr)
        X_te, y_te, gate_te, te_ents = _extract_all(records, cfg)
        print(f"  {len(X_te)}/{len(records)} extracted.", file=sys.stderr)
        n_tr, n_te = len(X_tr), len(X_te)
        mode = (
            f"cross-dataset  "
            f"train={os.path.basename(args.train_labels)} ({n_tr})"
            f"  eval={os.path.basename(args.labels)} ({n_te})"
        )
    else:
        # Original mode: single dataset split by base_utterance_id.
        print(f"Extracting features ({len(records)} files)...", file=sys.stderr)
        X, y, gate_preds, valid = _extract_all(records, cfg)
        print(f"  {len(valid)}/{len(records)} files extracted.", file=sys.stderr)
        tr_idx, te_idx = _make_split(valid, args.train_frac, args.seed)
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        gate_te    = gate_preds[te_idx]
        te_ents    = [valid[i] for i in te_idx]
        n_tr, n_te = len(X_tr), len(X_te)
        mode = f"split by base_utterance_id ({args.train_frac:.0%} train, seed={args.seed})"
        print(
            f"Train: {n_tr}"
            f"  (pos={int(y_tr.sum())} neg={n_tr - int(y_tr.sum())})"
            f"  Test: {n_te}"
            f"  (pos={int(y_te.sum())} neg={n_te - int(y_te.sum())})",
            file=sys.stderr,
        )

    # Scaled versions for logistic regression.
    scaler  = StandardScaler().fit(X_tr)
    X_tr_s  = scaler.transform(X_tr)
    X_te_s  = scaler.transform(X_te)

    # -----------------------------------------------------------------------
    # Train and evaluate (all 27 features)
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
        print(f"Training {name} (all features)...", file=sys.stderr)
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

    _print_report(all_results, n_tr, n_te, mode)

    # -----------------------------------------------------------------------
    # Ablation: DSP-only (24 features) vs all (27 features)
    # -----------------------------------------------------------------------
    print("Running ablation (DSP-only vs all features)...", file=sys.stderr)
    ablation_dsp: List[Dict] = []
    # Fresh instances — the model_specs clfs are already fitted on all features.
    dsp_specs = [
        ("logreg", LogisticRegression(C=1.0, solver="liblinear", max_iter=500, random_state=SEED), True),
        ("rf",     RandomForestClassifier(n_estimators=100, max_depth=6, min_samples_leaf=3, n_jobs=-1, random_state=SEED), False),
        ("gbt",    GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=SEED), False),
    ]
    for dsp_name, clf_dsp, use_scaled in dsp_specs:
        print(f"  {dsp_name} dsp...", file=sys.stderr)
        Xtr_d = (X_tr_s if use_scaled else X_tr)[:, :N_DSP_FEATURES]
        Xte_d = (X_te_s if use_scaled else X_te)[:, :N_DSP_FEATURES]
        clf_dsp.fit(Xtr_d, y_tr)
        yp_d  = clf_dsp.predict(Xte_d)
        ablation_dsp.append({
            "name":        f"{dsp_name}_dsp",
            "feature_set": "dsp_only",
            "metrics":     _eval_metrics(y_te, yp_d, te_ents),
        })

    # Print ablation comparison.
    abl_all = {r["name"]: r["metrics"] for r in all_results if r["name"] != "rule_gate"}
    abl_dsp = {r["name"].replace("_dsp", ""): r["metrics"] for r in ablation_dsp}
    print("\n--- Ablation: gate-decision fractions vs DSP-only ---")
    abl_hdr = (
        f"  {'Model':12s}  {'Features':9s}  {'FAR':>6}  {'FRR':>6}  "
        f"{'F1':>6}  {'MusicFA':>8}  {'CleanFR':>8}"
    )
    print(abl_hdr)
    print("  " + "-" * (len(abl_hdr) - 2))
    for mname in ("logreg", "rf", "gbt"):
        for feat_label, mdict in (("all (27)", abl_all), ("dsp (24)", abl_dsp)):
            m = mdict.get(mname)
            if m:
                print(
                    f"  {mname:12s}  {feat_label:9s}  {m['far']:>6.4f}  "
                    f"{m['frr']:>6.4f}  {m['f1']:>6.4f}  "
                    f"{m['music_fa']:>8d}  {m['clean_speech_fr']:>8d}"
                )
    print()

    # -----------------------------------------------------------------------
    # Threshold sweep (models with predict_proba)
    # -----------------------------------------------------------------------
    rule_metrics  = all_results[0]["metrics"]
    sweep_all:    List[Dict] = []
    ops_by_model: Dict[str, Dict] = {}
    for name, clf, use_scaled in model_specs:
        if not hasattr(clf, "predict_proba"):
            continue
        Xte   = X_te_s if use_scaled else X_te
        proba = clf.predict_proba(Xte)[:, 1]
        rows  = _sweep_thresholds(name, y_te, proba, te_ents)
        sweep_all.extend(rows)
        ops_by_model[name] = _recommend_operating_points(rows, rule_metrics["frr"])

    if ops_by_model:
        _print_ops_summary(ops_by_model, rule_metrics)

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    if args.out:
        out_dir = args.out if os.path.isabs(args.out) else os.path.join(_repo, args.out)
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # model_comparison.json
        mc_path = os.path.join(out_dir, f"model_comparison_{ts}.json")
        with open(mc_path, "w") as fh:
            json.dump(
                {
                    "ts":            ts,
                    "n_records":     len(records),
                    "train_labels":  args.train_labels,
                    "train_n":       n_tr,
                    "test_n":        n_te,
                    "mode":          mode,
                    "train_frac":    args.train_frac,
                    "seed":          args.seed,
                    "chunk_sec":     CHUNK_SEC,
                    "feature_names": FEATURE_NAMES,
                    "models":        all_results,
                    "ablation":      ablation_dsp,
                },
                fh,
                indent=2,
            )
        print(f"Saved: {mc_path}", file=sys.stderr)

        # feature_importance.json
        fi_rows = [
            {"model": r["name"], "importances": r["feature_importances"]}
            for r in all_results if "feature_importances" in r
        ]
        if fi_rows:
            fi_path = os.path.join(out_dir, f"feature_importance_{ts}.json")
            with open(fi_path, "w") as fh:
                json.dump(fi_rows, fh, indent=2)
            print(f"Saved: {fi_path}", file=sys.stderr)

        # threshold_sweep.tsv
        if sweep_all:
            tsv_path = os.path.join(out_dir, f"threshold_sweep_{ts}.tsv")
            with open(tsv_path, "w") as fh:
                fh.write("model\tthreshold\tfar\tfrr\tprecision\trecall\tf1\tmusic_fa\tclean_speech_fr\n")
                for row in sweep_all:
                    fh.write(
                        f"{row['model']}\t{row['threshold']}\t"
                        f"{row['far']}\t{row['frr']}\t"
                        f"{row['precision']}\t{row['recall']}\t{row['f1']}\t"
                        f"{row['music_fa']}\t{row['clean_speech_fr']}\n"
                    )
            print(f"Saved: {tsv_path}", file=sys.stderr)

        # recommended_operating_points.json
        if ops_by_model:
            ops_path = os.path.join(out_dir, f"recommended_operating_points_{ts}.json")
            with open(ops_path, "w") as fh:
                json.dump(
                    {
                        "ts":        ts,
                        "mode":      mode,
                        "rule_gate": rule_metrics,
                        "models": {
                            n: {
                                "balanced":     ops["balanced"],
                                "conservative": ops["conservative"],
                            }
                            for n, ops in ops_by_model.items()
                        },
                    },
                    fh,
                    indent=2,
                )
            print(f"Saved: {ops_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
