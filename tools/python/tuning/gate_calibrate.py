#!/usr/bin/env python3
"""
gate_calibrate.py
Label-aware threshold sweep for the rule-based DSP gate.

Reads a calibration JSON produced by gate_eval.py (which contains pre-computed
chunk metrics and file-level labels). Re-simulates gate decisions at every grid
point without re-loading audio. Scores each config by:

  cost = FAR + alpha * FRR

  FAR (false accept rate)  = files where should_transcribe=no  but gate PASS
  FRR (false reject rate)  = files where should_transcribe=yes but gate FAIL
  alpha (default 0.5): FRR is weighted slightly less because missing some
    transcriptions is less harmful than flooding ASR with noise.

Swept thresholds (others held at baseline):
  rms_min              -- BORDERLINE low-energy floor
  spectral_flatness_max -- FAIL: stationary noise rejection ceiling
  min_band_mid          -- FAIL: minimum mid-band (500-4000 Hz) speech energy
  max_silence_ratio     -- FAIL: maximum tolerated silence fraction

Output:
  benchmarks/results/gate_calibration/calibrated_TIMESTAMP.json
  benchmarks/results/gate_calibration/sweep_TIMESTAMP.tsv

Usage:
    python tools/python/tuning/gate_calibrate.py \\
        benchmarks/results/gate_calibration/baseline_TIMESTAMP.json
    python tools/python/tuning/gate_calibrate.py baseline.json --alpha 0.5
    python tools/python/tuning/gate_calibrate.py baseline.json --csv > sweep.csv
"""

import argparse
import datetime
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Gate simulation — mirrors evaluate_chunk() in gate.cpp.
# Field names match the schema defined in schema.instructions.md and produced
# by gate_eval.py chunk records.
# ---------------------------------------------------------------------------

def _simulate_chunk(m: Dict, cfg: Dict) -> str:
    """Simulate a gate decision from pre-computed chunk metrics.

    Must mirror gate.cpp evaluate_chunk() exactly. Field names used here must
    match what gate_eval.py writes into calibration JSON chunk records.
    """
    if m["rms"] < cfg["rms_borderline_min"]:
        return "FAIL"
    if m["silence_ratio"] > cfg["max_silence_ratio"]:
        return "FAIL"
    if m["clipping_ratio"] > cfg["max_clipping_ratio"]:
        return "FAIL"
    if m["active_frac"] < cfg["min_active_frame_frac"]:
        return "FAIL"
    if m["flatness"] > cfg["spectral_flatness_max"]:
        return "FAIL"
    if m["flatness"] > cfg["spectral_flatness_warn"] and m["zcr"] > cfg["zcr_max_noise"]:
        return "FAIL"
    if m["band_mid"] < cfg["min_band_mid"]:
        return "FAIL"
    if m["band_high"] > cfg["max_band_high"]:
        return "FAIL"
    if m["rms"] < cfg["rms_min"]:
        return "BORDERLINE"
    if m["flatness"] > cfg["spectral_flatness_warn"]:
        return "BORDERLINE"
    return "PASS"


def _file_decision(chunk_decisions: List[str]) -> str:
    if "PASS" in chunk_decisions:
        return "PASS"
    if "BORDERLINE" in chunk_decisions:
        return "BORDERLINE"
    return "FAIL"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_config(
    file_records: List[Dict],
    cfg: Dict,
    detailed: bool = False,
) -> Dict:
    """Evaluate a threshold config against all labeled files.

    Returns a dict with: n_pass, n_borderline, n_fail, n_fa, n_fr,
    far, frr, accept_rate.

    When detailed=True also returns: by_label confusion tables,
    false_accept_examples, false_reject_examples.

    Audio concept: calibration finds the config that minimises classification
    error on known-labeled audio without any learned model. This is threshold
    calibration, not training.
    """
    n_should_yes = sum(1 for r in file_records if r.get("should_transcribe") == "yes")
    n_should_no  = sum(1 for r in file_records if r.get("should_transcribe") == "no")
    n_total      = len(file_records)

    n_pass = n_borderline = n_fail = 0
    n_fa = n_fr = 0

    # Per-label confusion: label -> {PASS: n, BORDERLINE: n, FAIL: n, total: n}
    by_label: Dict[str, Dict] = {}
    fa_examples: List[str] = []
    fr_examples: List[str] = []

    for r in file_records:
        decs = [_simulate_chunk(c, cfg) for c in r.get("chunks", [])]
        dec  = _file_decision(decs) if decs else "FAIL"

        if dec == "PASS":
            n_pass += 1
        elif dec == "BORDERLINE":
            n_borderline += 1
        else:
            n_fail += 1

        st = r.get("should_transcribe", "")
        if st == "no"  and dec == "PASS":
            n_fa += 1
            if detailed:
                fa_examples.append(os.path.basename(r.get("path", "")))
        if st == "yes" and dec == "FAIL":
            n_fr += 1
            if detailed:
                fr_examples.append(os.path.basename(r.get("path", "")))

        if detailed:
            lbl = r.get("label", "")
            if lbl not in by_label:
                by_label[lbl] = {"PASS": 0, "BORDERLINE": 0, "FAIL": 0, "total": 0}
            by_label[lbl][dec] += 1
            by_label[lbl]["total"] += 1

    far = n_fa / max(n_should_no, 1)
    frr = n_fr / max(n_should_yes, 1)
    accept_rate = (n_pass + n_borderline) / max(n_total, 1)

    result: Dict = {
        "n_pass":       n_pass,
        "n_borderline": n_borderline,
        "n_fail":       n_fail,
        "n_fa":         n_fa,
        "n_fr":         n_fr,
        "far":          far,
        "frr":          frr,
        "accept_rate":  accept_rate,
    }
    if detailed:
        result["by_label"]              = by_label
        result["false_accept_examples"] = fa_examples
        result["false_reject_examples"] = fr_examples
    return result


# ---------------------------------------------------------------------------
# Sweep grid
# ---------------------------------------------------------------------------

# rms_min: BORDERLINE energy floor.
# Raising it converts more low-energy speech to BORDERLINE; lowering it lets
# near-silent frames through.
RMS_MIN_GRID         = [0.001, 0.002, 0.003, 0.005, 0.008, 0.010]

# spectral_flatness_max: FAIL ceiling for noise-like flatness.
# Lower values reject more noise but risk rejecting reverberant speech.
FLATNESS_MAX_GRID    = [0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

# min_band_mid: FAIL floor for mid-band (500-4000 Hz) energy fraction.
# Raising it rejects more low-frequency-only content (HVAC hum); too high
# rejects legitimate speech with energy redistributed by noise.
MIN_BAND_MID_GRID    = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]

# max_silence_ratio: FAIL ceiling for fraction of silent samples.
# Lowering it rejects chunks with sparse speech more aggressively.
SILENCE_MAX_GRID     = [0.80, 0.85, 0.90, 0.95]


def grid_dims() -> Dict:
    """Return the actual sweep grid dimensions, computed from the grid lists."""
    return {
        "rms_min":      {"values": RMS_MIN_GRID,      "n": len(RMS_MIN_GRID)},
        "flatness_max": {"values": FLATNESS_MAX_GRID,  "n": len(FLATNESS_MAX_GRID)},
        "min_band_mid": {"values": MIN_BAND_MID_GRID,  "n": len(MIN_BAND_MID_GRID)},
        "silence_max":  {"values": SILENCE_MAX_GRID,   "n": len(SILENCE_MAX_GRID)},
        "total":        len(RMS_MIN_GRID) * len(FLATNESS_MAX_GRID) *
                        len(MIN_BAND_MID_GRID) * len(SILENCE_MAX_GRID),
    }


def run_sweep(
    file_records: List[Dict],
    base_cfg: Dict,
    alpha: float,
) -> Tuple[List[Dict], Optional[Dict]]:
    """Grid sweep over 4 thresholds. Returns (all_rows, best_cfg)."""
    rows: List[Dict] = []
    best_cfg:  Optional[Dict] = None
    best_cost: float = float("inf")

    dims    = grid_dims()
    n_total = dims["total"]
    done    = 0

    for rms_min in RMS_MIN_GRID:
        for flatness_max in FLATNESS_MAX_GRID:
            for min_band_mid in MIN_BAND_MID_GRID:
                for silence_max in SILENCE_MAX_GRID:
                    cfg = dict(base_cfg)
                    cfg["rms_min"]               = rms_min
                    cfg["spectral_flatness_max"] = flatness_max
                    # flatness_warn is derived as 80% of flatness_max, same as
                    # the relationship in the default config (0.72 = 0.90*0.80).
                    cfg["spectral_flatness_warn"] = round(flatness_max * 0.80, 4)
                    cfg["min_band_mid"]           = min_band_mid
                    cfg["max_silence_ratio"]      = silence_max

                    s    = score_config(file_records, cfg)
                    cost = s["far"] + alpha * s["frr"]

                    row: Dict = {
                        "rms_min":      rms_min,
                        "flatness_max": flatness_max,
                        "min_band_mid": min_band_mid,
                        "silence_max":  silence_max,
                        "far":          s["far"],
                        "frr":          s["frr"],
                        "accept_rate":  s["accept_rate"],
                        "n_pass":       s["n_pass"],
                        "n_borderline": s["n_borderline"],
                        "n_fail":       s["n_fail"],
                        "n_fa":         s["n_fa"],
                        "n_fr":         s["n_fr"],
                        "cost":         cost,
                    }
                    rows.append(row)

                    if cost < best_cost:
                        best_cost = cost
                        best_cfg  = dict(cfg)

                    done += 1
                    print(
                        f"\r  sweep {done}/{n_total}",
                        end="", file=sys.stderr, flush=True,
                    )

    print(file=sys.stderr)
    return rows, best_cfg


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------

def _fmt_score(label: str, s: Dict) -> None:
    n = max(s["n_pass"] + s["n_borderline"] + s["n_fail"], 1)
    print(f"\n{label}")
    print(
        f"  PASS={s['n_pass']} ({s['n_pass']/n:.1%})  "
        f"BORDERLINE={s['n_borderline']} ({s['n_borderline']/n:.1%})  "
        f"FAIL={s['n_fail']} ({s['n_fail']/n:.1%})"
    )
    print(f"  FAR (no ->PASS)  : {s['n_fa']:3d}  {s['far']:.4f}")
    print(f"  FRR (yes->FAIL)  : {s['n_fr']:3d}  {s['frr']:.4f}")
    print(f"  Accept rate      : {s['accept_rate']:.4f}")


def _print_config(cfg: Dict) -> None:
    """Print recommended thresholds in both human-readable and C++ initializer forms."""
    print()
    print("Recommended GateConfig values:")
    fields = [
        ("rms_borderline_min",    "rms_borderline_min"),
        ("rms_min",               "rms_min"),
        ("max_silence_ratio",     "max_silence_ratio"),
        ("max_clipping_ratio",    "max_clipping_ratio"),
        ("min_active_frame_frac", "min_active_frame_frac"),
        ("spectral_flatness_max", "spectral_flatness_max"),
        ("spectral_flatness_warn","spectral_flatness_warn"),
        ("min_band_mid",          "min_band_mid"),
        ("max_band_high",         "max_band_high"),
    ]
    for key, _ in fields:
        print(f"  {key:28s} = {cfg[key]}")

    print()
    print("C++ struct initializer (paste into gate.h GateConfig defaults):")
    print("  // calibrated thresholds — gate_calibrate.py output")
    for key, _ in fields:
        print(f"  cfg.{key:<28s} = {cfg[key]};")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "baseline",
        help="Baseline JSON from gate_eval.py",
    )
    ap.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="FRR weight in cost=FAR+alpha*FRR (default: 0.5)",
    )
    ap.add_argument(
        "--csv",
        action="store_true",
        help="Print full sweep table as CSV instead of formatted output",
    )
    args = ap.parse_args()

    with open(args.baseline) as fh:
        artifact = json.load(fh)

    file_records: List[Dict] = artifact["files"]
    base_cfg:     Dict       = artifact["meta"]["config"]

    print(
        f"Loaded {len(file_records)} file records from {args.baseline}",
        file=sys.stderr,
    )

    # Baseline score using current defaults
    baseline_score = score_config(file_records, base_cfg, detailed=True)
    _fmt_score("BASELINE (current GateConfig defaults)", baseline_score)

    dims   = grid_dims()
    n_grid = dims["total"]
    print(
        f"\nSweeping "
        f"{dims['rms_min']['n']} x {dims['flatness_max']['n']} x "
        f"{dims['min_band_mid']['n']} x {dims['silence_max']['n']} = {n_grid} grid points "
        f"(alpha={args.alpha})...",
        file=sys.stderr,
    )

    rows, best_cfg = run_sweep(file_records, base_cfg, args.alpha)

    if args.csv:
        cols = [
            "rms_min", "flatness_max", "min_band_mid", "silence_max",
            "far", "frr", "accept_rate", "n_pass", "n_borderline", "n_fail",
            "n_fa", "n_fr", "cost",
        ]
        print(",".join(cols))
        for r in rows:
            vals = []
            for c in cols:
                v = r[c]
                vals.append(f"{v:.6f}" if isinstance(v, float) else str(v))
            print(",".join(vals))
        return

    # Human-readable: top-10 operating points sorted by cost
    rows_sorted = sorted(rows, key=lambda r: r["cost"])

    print(f"\nTop 10 operating points  (cost = FAR + {args.alpha} * FRR):")
    hdr = (
        f"  {'rms_min':>8}  {'flat_max':>8}  {'bnd_mid':>7}  {'sil_max':>7}  "
        f"{'FAR':>7}  {'FRR':>7}  {'AR':>7}  {'FA':>4}  {'FR':>4}  {'cost':>8}"
    )
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in rows_sorted[:10]:
        print(
            f"  {r['rms_min']:>8.4f}  {r['flatness_max']:>8.4f}  "
            f"{r['min_band_mid']:>7.4f}  {r['silence_max']:>7.4f}  "
            f"{r['far']:>7.4f}  {r['frr']:>7.4f}  {r['accept_rate']:>7.4f}  "
            f"{r['n_fa']:>4d}  {r['n_fr']:>4d}  {r['cost']:>8.5f}"
        )

    calibrated_score: Optional[Dict] = None
    if best_cfg:
        calibrated_score = score_config(file_records, best_cfg, detailed=True)
        _fmt_score("CALIBRATED (recommended operating point)", calibrated_score)
        _print_config(best_cfg)

        far_d = calibrated_score["far"] - baseline_score["far"]
        frr_d = calibrated_score["frr"] - baseline_score["frr"]
        ar_d  = calibrated_score["accept_rate"] - baseline_score["accept_rate"]
        print()
        print(
            f"Delta vs baseline:  FAR {far_d:+.4f}   FRR {frr_d:+.4f}"
            f"   accept_rate {ar_d:+.4f}"
        )

        # Per-label confusion at calibrated operating point
        print()
        print("Per-label confusion (calibrated):")
        for lbl, vals in sorted(calibrated_score.get("by_label", {}).items()):
            t = max(vals["total"], 1)
            print(
                f"  {lbl:35s}  total={vals['total']:4d}  "
                f"PASS={vals['PASS']:3d} ({vals['PASS']/t:5.1%})  "
                f"BORDER={vals['BORDERLINE']:3d} ({vals['BORDERLINE']/t:5.1%})  "
                f"FAIL={vals['FAIL']:3d} ({vals['FAIL']/t:5.1%})"
            )

    # Save outputs
    ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.dirname(os.path.abspath(args.baseline))
    dims    = grid_dims()

    # Calibrated JSON
    calibrated_path = os.path.join(out_dir, f"calibrated_{ts}.json")
    out_artifact = {
        "meta": {
            "timestamp":     ts,
            "baseline_file": args.baseline,
            "alpha":         args.alpha,
            "grid_dims":     dims,
            "base_config":   base_cfg,
            "best_config":   best_cfg,
        },
        "baseline_score":   baseline_score,
        "calibrated_score": calibrated_score,
        "top10":            rows_sorted[:10],
    }
    with open(calibrated_path, "w") as fh:
        json.dump(out_artifact, fh, indent=2)

    # Recommended GateConfig JSON — standalone for easy diffing
    if best_cfg:
        rec_path = os.path.join(out_dir, "recommended_gate_config.json")
        with open(rec_path, "w") as fh:
            json.dump(
                {
                    "source":    f"gate_calibrate.py sweep {ts}",
                    "alpha":     args.alpha,
                    "cost":      round(
                        calibrated_score["far"] + args.alpha * calibrated_score["frr"], 6
                    ),
                    "far":       calibrated_score["far"],
                    "frr":       calibrated_score["frr"],
                    "accept_rate": calibrated_score["accept_rate"],
                    "config":    best_cfg,
                },
                fh,
                indent=2,
            )
        print(f"Recommended config : {rec_path}", file=sys.stderr)

    # Sweep TSV — full grid, for offline analysis
    sweep_path = os.path.join(out_dir, f"sweep_{ts}.tsv")
    tsv_cols = [
        "rms_min", "flatness_max", "min_band_mid", "silence_max",
        "far", "frr", "accept_rate", "n_fa", "n_fr", "cost",
    ]
    with open(sweep_path, "w") as fh:
        fh.write("\t".join(tsv_cols) + "\n")
        for r in rows:
            fh.write(
                "\t".join(
                    f"{r[c]:.6f}" if isinstance(r[c], float) else str(r[c])
                    for c in tsv_cols
                ) + "\n"
            )

    print(f"Calibrated JSON    : {calibrated_path}", file=sys.stderr)
    print(f"Sweep TSV          : {sweep_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
