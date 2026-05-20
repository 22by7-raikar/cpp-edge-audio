#!/usr/bin/env python3
"""
gate_scene_policy.py
Evaluate admission policies combining the rule-based gate with scene classification.

Scene classification mirrors runtime/cpp/src/scene/scene.cpp classify().
It uses the pre-computed chunk metrics already in the baseline JSON from gate_eval.py.

Admission policies evaluated:
  A  gate_only
       accept = file_decision in {PASS, BORDERLINE}
  B  scene_only
       accept = any chunk scene in {SPEECH, MIXED, UNKNOWN}
  C  gate_and_scene_skip_music_silence
       accept = gate PASS/BORDER  AND  file scene is not MUSIC/SILENCE
  D  gate_and_scene_skip_music_silence_noise
       accept = gate PASS/BORDER  AND  file scene is not MUSIC/SILENCE/NOISE

Policy rationale:
  The gate baseline shows FAR=0.425, mostly music false accepts.
  Music is tonal (low flatness) and not rejected by the spectral flatness checks.
  Scene classification detects music via low flatness + significant low+high band
  energy, which is a different combination from the gate's hard thresholds.
  Policy C/D test whether adding a scene override reduces music FAs without
  increasing speech FRs.

  "Scene-only" policy B is included to show what would happen if we abandoned
  the gate entirely and only used scene context. It is NOT a recommended
  deployment policy — it has no energy or silence checks.

Output:
  - Per-policy report to stdout
  - policy_report_TIMESTAMP.json to --out dir

Usage:
    python tools/python/eval/gate_scene_policy.py \\
        benchmarks/results/gate_calibration/baseline_TIMESTAMP.json
    python tools/python/eval/gate_scene_policy.py baseline.json --out dir/
"""

import argparse
import datetime
import json
import math
import os
import sys
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Scene classification — mirrors scene.cpp classify()
# Default SceneConfig matches scene.h defaults exactly.
# ---------------------------------------------------------------------------

DEFAULT_SCENE_CONFIG: Dict = {
    # SILENCE
    "silence_rms_max":    0.003,
    "silence_active_max": 0.15,
    # NOISE: high spectral flatness = broadband noise
    "noise_flatness_min": 0.65,
    # MUSIC: tonal (low flatness) with energy in both low and high bands
    "music_flatness_max":  0.45,
    "music_band_low_min":  0.18,
    "music_band_high_min": 0.08,
    # SPEECH: mid-band dominant, centroid in voice range
    "speech_centroid_min": 200.0,
    "speech_centroid_max": 5000.0,
    "speech_band_mid_dom": 0.40,
    # MIXED_SPEECH_NOISE: speech-like but elevated flatness
    "mixed_flatness_min":  0.40,
}

# High-confidence threshold for conservative music rejection.
# Chunks classified as MUSIC only trigger a skip decision when the classifier
# confidence exceeds this value, reducing false rejects on voiced speech that
# borderline satisfies the MUSIC rule (low flatness + some low+high band energy).
HIGH_CONF: float = 0.7

# Stricter music detection — reduces clean speech mis-classified as MUSIC.
# Raising music_band_low_min from 0.18 to 0.25 requires more bass energy
# before firing the MUSIC rule. Male voices have band_low ~0.10-0.24;
# bass guitar / bass drum typically exceeds 0.25.
# Raising music_band_high_min from 0.08 to 0.12 similarly reduces
# false positives from speech with prominent sibilant consonants.
STRICT_MUSIC_CFG: Dict = {
    **DEFAULT_SCENE_CONFIG,
    "music_band_low_min":  0.25,
    "music_band_high_min": 0.12,
}

# Scene label strings — must match scene_label_str() in scene.cpp
SILENCE = "SILENCE"
NOISE   = "NOISE"
MUSIC   = "MUSIC"
SPEECH  = "SPEECH"
MIXED   = "MIXED"
UNKNOWN = "UNKNOWN"


def soft_conf(val: float, threshold: float, spread: float, direction: int) -> float:
    """Mirrors C++ soft_conf(). Returns smooth confidence in [0, 1].

    direction=+1: higher value => more confident (e.g. flatness above noise thresh)
    direction=-1: lower value  => more confident (e.g. rms below silence thresh)
    """
    raw = direction * (val - threshold) / max(spread, 1e-9)
    return min(1.0, max(0.0, 0.5 + 0.5 * math.tanh(raw * 3.0)))


def classify_chunk_with_conf(m: Dict, cfg: Dict) -> Tuple[str, float]:
    """Scene classification with confidence score. Returns (label, confidence).

    Mirrors scene.cpp classify() including the confidence computation.
    confidence is a heuristic score in [0, 1]: how strongly the features match
    the winning rule. Used to implement conservative high-confidence-only skips.
    """
    rms      = m.get("rms", 0.0)
    active   = m.get("active_frac", 0.0)
    flatness = m.get("flatness", 0.0)
    centroid = m.get("centroid_hz", 0.0)
    band_low = m.get("band_low", 0.0)
    band_mid = m.get("band_mid", 0.0)
    band_high= m.get("band_high", 0.0)

    # 1. SILENCE
    if rms < cfg["silence_rms_max"] or active < cfg["silence_active_max"]:
        conf = soft_conf(rms, cfg["silence_rms_max"], cfg["silence_rms_max"], -1)
        return SILENCE, conf

    # No spectral features computed (gate short-circuited before FFT).
    if flatness == 0.0 and centroid == 0.0 and band_mid == 0.0:
        return UNKNOWN, 0.0

    # 2. NOISE — spectrally flat
    if flatness >= cfg["noise_flatness_min"]:
        conf = soft_conf(flatness, cfg["noise_flatness_min"],
                         1.0 - cfg["noise_flatness_min"], +1)
        return NOISE, conf

    # 3. MUSIC — tonal with energy in both low and high bands
    is_tonal        = flatness  < cfg["music_flatness_max"]
    has_low_energy  = band_low  >= cfg["music_band_low_min"]
    has_high_energy = band_high >= cfg["music_band_high_min"]
    if is_tonal and has_low_energy and has_high_energy:
        s_tonal = soft_conf(flatness,  cfg["music_flatness_max"],  cfg["music_flatness_max"],  -1)
        s_low   = soft_conf(band_low,  cfg["music_band_low_min"],  cfg["music_band_low_min"],  +1)
        s_high  = soft_conf(band_high, cfg["music_band_high_min"], cfg["music_band_high_min"], +1)
        conf = (s_tonal * s_low * s_high) ** (1.0 / 3.0)  # geometric mean, mirrors cbrt()
        return MUSIC, conf

    # 4. SPEECH — centroid in voice range, mid-band dominant, low flatness
    centroid_ok = cfg["speech_centroid_min"] <= centroid <= cfg["speech_centroid_max"]
    mid_dom     = band_mid >= cfg["speech_band_mid_dom"]
    flat_ok     = flatness < cfg["mixed_flatness_min"]
    if centroid_ok and mid_dom and flat_ok:
        conf = soft_conf(band_mid, cfg["speech_band_mid_dom"],
                         1.0 - cfg["speech_band_mid_dom"], +1)
        return SPEECH, conf

    # 5. MIXED — speech-like but elevated flatness
    if centroid_ok and mid_dom:
        conf = soft_conf(flatness, cfg["mixed_flatness_min"],
                         cfg["noise_flatness_min"] - cfg["mixed_flatness_min"], +1)
        return MIXED, conf

    return UNKNOWN, 0.0


def classify_chunk(m: Dict, cfg: Dict) -> str:
    """Backward-compatible wrapper — returns scene label only."""
    return classify_chunk_with_conf(m, cfg)[0]


def file_scene_label(chunk_labels: List[str]) -> str:
    """Aggregate chunk scene labels to a file-level scene.

    Priority:
      SPEECH > MIXED > UNKNOWN > MUSIC > NOISE > SILENCE

    Audio concept: if any chunk in the file is classified as speech,
    the file is considered speech-containing. Music/noise labels are only
    assigned at the file level when no speech was found in any chunk.
    """
    for label in (SPEECH, MIXED, UNKNOWN):
        if label in chunk_labels:
            return label
    for label in (MUSIC, NOISE, SILENCE):
        if label in chunk_labels:
            return label
    return UNKNOWN


# ---------------------------------------------------------------------------
# Gate decision helpers (same as gate_eval.py / gate_calibrate.py)
# ---------------------------------------------------------------------------

def _gate_accepts(file_decision: str) -> bool:
    return file_decision in ("PASS", "BORDERLINE")


# ---------------------------------------------------------------------------
# Policy definitions
# ---------------------------------------------------------------------------

POLICIES = {
    "A_gate_only": {
        "description": "Gate PASS/BORDERLINE (baseline, no scene)",
    },
    "B_gate_high_conf_music_skip": {
        "description":
            f"Gate PASS/BORDER AND NOT (scene=MUSIC with conf >= {HIGH_CONF})",
    },
    "C_gate_high_conf_music_silence_skip": {
        "description":
            f"Gate PASS/BORDER AND NOT (MUSIC conf>={HIGH_CONF} OR file_scene=SILENCE)",
    },
    "D_gate_strict_music_skip": {
        "description":
            "Gate PASS/BORDER AND NOT (MUSIC under strict thresholds: "
            "band_low>=0.25, band_high>=0.12)",
    },
}


# Accept functions take a pre-built entry dict with scene and confidence fields.
def _accept_A(e: Dict) -> bool:
    return _gate_accepts(e["file_gate"])


def _accept_B(e: Dict) -> bool:
    # Reject only when MUSIC confidence is high — avoids penalising speech that
    # borderline triggers the MUSIC rule with low confidence.
    if not _gate_accepts(e["file_gate"]):
        return False
    return e["max_music_conf"] < HIGH_CONF


def _accept_C(e: Dict) -> bool:
    # Like B, but also rejects files that the classifier is certain are silent.
    # SILENCE here means file_scene_label() returned SILENCE, i.e. every chunk
    # was SILENCE — a truly quiet file. Silent files are safe to reject hard.
    if not _gate_accepts(e["file_gate"]):
        return False
    if e["max_music_conf"] >= HIGH_CONF:
        return False
    if e["file_scene"] == SILENCE:
        return False
    return True


def _accept_D(e: Dict) -> bool:
    # Stricter music thresholds (band_low_min=0.25, band_high_min=0.12).
    # Under these thresholds fewer speech files are mis-classified as MUSIC,
    # so a hard reject on MUSIC at the file level becomes safer.
    if not _gate_accepts(e["file_gate"]):
        return False
    return e["file_scene_strict"] != MUSIC


ACCEPT_FNS = {
    "A_gate_only":                         _accept_A,
    "B_gate_high_conf_music_skip":         _accept_B,
    "C_gate_high_conf_music_silence_skip": _accept_C,
    "D_gate_strict_music_skip":            _accept_D,
}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_policies(
    file_records: List[Dict],
    scene_cfg: Dict,
) -> Dict[str, Dict]:
    """Classify every chunk's scene, then evaluate each policy.

    Returns dict: policy_name -> {n_accept, n_reject, far, frr, by_label, ...}
    """
    n_should_yes = sum(1 for r in file_records if r.get("should_transcribe") == "yes")
    n_should_no  = sum(1 for r in file_records if r.get("should_transcribe") == "no")

    # Per-file scene classification — one pass, results reused across all policies.
    file_entries: List[Dict] = []
    scene_dist: Dict[str, int] = {}
    for r in file_records:
        chunks = r.get("chunks", [])

        # Default config — classify with confidence scores.
        def_results        = [classify_chunk_with_conf(c, scene_cfg) for c in chunks]
        chunk_scene_labels = [x[0] for x in def_results]
        chunk_scene_confs  = [x[1] for x in def_results]
        f_scene            = file_scene_label(chunk_scene_labels)
        # Max confidence among MUSIC chunks (0 if no MUSIC chunks).
        max_music_conf = max(
            (conf for lbl, conf in zip(chunk_scene_labels, chunk_scene_confs)
             if lbl == MUSIC),
            default=0.0,
        )

        for s in chunk_scene_labels:
            scene_dist[s] = scene_dist.get(s, 0) + 1

        # Strict music config — for policy D.
        str_results         = [classify_chunk_with_conf(c, STRICT_MUSIC_CFG) for c in chunks]
        chunk_scenes_strict = [x[0] for x in str_results]
        file_scene_strict   = file_scene_label(chunk_scenes_strict)

        file_entries.append({
            "record":             r,
            "chunk_scenes":       chunk_scene_labels,
            "chunk_confs":        chunk_scene_confs,
            "file_scene":         f_scene,
            "max_music_conf":     max_music_conf,
            "chunk_scenes_strict": chunk_scenes_strict,
            "file_scene_strict":  file_scene_strict,
            "file_gate":          r.get("file_decision", "FAIL"),
            "should_transcribe":  r.get("should_transcribe", ""),
            "label":              r.get("label", ""),
        })

    # Evaluate each policy
    results: Dict[str, Dict] = {}
    for pname, accept_fn in ACCEPT_FNS.items():
        n_accept = 0
        n_reject = 0
        n_fa = 0
        n_fr = 0
        by_label: Dict[str, Dict] = {}
        fa_examples: List[str] = []
        fr_examples: List[str] = []
        music_fa_count = 0

        for e in file_entries:
            accepted = accept_fn(e)
            lbl  = e["label"]
            st   = e["should_transcribe"]
            path = e["record"].get("path", "")

            if accepted:
                n_accept += 1
            else:
                n_reject += 1

            if st == "no"  and     accepted: n_fa += 1
            if st == "yes" and not accepted: n_fr += 1

            if lbl not in by_label:
                by_label[lbl] = {"accept": 0, "reject": 0, "total": 0}
            by_label[lbl]["accept" if accepted else "reject"] += 1
            by_label[lbl]["total"] += 1

            if st == "no" and accepted:
                fa_examples.append({
                    "path":       os.path.basename(path),
                    "label":      lbl,
                    "scene":      e["file_scene"],
                    "gate":       e["file_gate"],
                    "music_conf": round(e["max_music_conf"], 3),
                })
                if lbl == "music":
                    music_fa_count += 1
            if st == "yes" and not accepted:
                fr_examples.append({
                    "path":  os.path.basename(path),
                    "label": lbl,
                    "scene": e["file_scene"],
                    "gate":  e["file_gate"],
                })

        n_total = n_accept + n_reject
        far = n_fa / max(n_should_no, 1)
        frr = n_fr / max(n_should_yes, 1)
        accept_rate = n_accept / max(n_total, 1)

        results[pname] = {
            "description":        POLICIES[pname]["description"],
            "n_accept":           n_accept,
            "n_reject":           n_reject,
            "n_fa":               n_fa,
            "n_fr":               n_fr,
            "far":                far,
            "frr":                frr,
            "accept_rate":        accept_rate,
            "music_fa_count":     music_fa_count,
            "by_label":           by_label,
            "false_accept_examples": fa_examples[:20],
            "false_reject_examples": fr_examples[:20],
        }

    return results, scene_dist


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def print_scene_dist(scene_dist: Dict[str, int]) -> None:
    total = sum(scene_dist.values())
    print()
    print("Chunk-level scene label distribution:")
    for label in (SPEECH, MIXED, MUSIC, NOISE, SILENCE, UNKNOWN):
        n = scene_dist.get(label, 0)
        print(f"  {label:20s}  {n:5d}  ({n/max(total,1):5.1%})")


def print_policy_report(pname: str, r: Dict, n_total_files: int) -> None:
    print()
    sep = "-" * 68
    print(sep)
    print(f"Policy {pname}")
    print(f"  {r['description']}")
    print(sep)
    n = max(n_total_files, 1)
    print(
        f"  Accept : {r['n_accept']:4d} ({r['n_accept']/n:5.1%})  "
        f"Reject : {r['n_reject']:4d} ({r['n_reject']/n:5.1%})"
    )
    print(f"  FAR (no ->accept)  : {r['n_fa']:3d}  {r['far']:.4f}")
    print(f"  FRR (yes->reject)  : {r['n_fr']:3d}  {r['frr']:.4f}")
    print(f"  Accept rate        : {r['accept_rate']:.4f}")
    print(f"  Music FA count     : {r['music_fa_count']}")

    print()
    print("  By label:")
    for lbl, vals in sorted(r["by_label"].items()):
        t = max(vals["total"], 1)
        pct_a = vals["accept"] / t
        print(
            f"    {lbl:35s}  total={vals['total']:4d}  "
            f"accept={vals['accept']:3d} ({pct_a:5.1%})  "
            f"reject={vals['reject']:3d} ({1-pct_a:5.1%})"
        )

    fae = r.get("false_accept_examples", [])
    fre = r.get("false_reject_examples", [])
    if fae:
        print()
        print(f"  False accepts ({len(fae)}):")
        for ex in fae[:8]:
            print(f"    {ex['path']:45s}  label={ex['label']:30s}  scene={ex['scene']}")
        if len(fae) > 8:
            print(f"    ... ({len(fae)-8} more)")
    if fre:
        print()
        print(f"  False rejects ({len(fre)}):")
        for ex in fre[:8]:
            print(f"    {ex['path']:45s}  label={ex['label']:30s}  scene={ex['scene']}")
        if len(fre) > 8:
            print(f"    ... ({len(fre)-8} more)")


def print_comparison_table(policy_results: Dict[str, Dict]) -> None:
    print()
    print("=" * 68)
    print("POLICY COMPARISON SUMMARY")
    print("=" * 68)
    hdr = (
        f"  {'Policy':46s}  {'FAR':>7}  {'FRR':>7}  {'FA':>4}  {'FR':>4}"
        f"  {'AR':>7}  {'Music FA':>8}"
    )
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for pname, r in policy_results.items():
        print(
            f"  {pname:46s}  {r['far']:>7.4f}  {r['frr']:>7.4f}  "
            f"{r['n_fa']:>4d}  {r['n_fr']:>4d}  "
            f"{r['accept_rate']:>7.4f}  {r['music_fa_count']:>8d}"
        )
    print()


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
        "--out",
        default=None,
        help="Output directory for policy_report JSON",
    )
    args = ap.parse_args()

    with open(args.baseline) as fh:
        artifact = json.load(fh)

    file_records: List[Dict] = artifact["files"]
    n_total = len(file_records)

    print(
        f"Loaded {n_total} file records from {args.baseline}",
        file=sys.stderr,
    )

    scene_cfg = dict(DEFAULT_SCENE_CONFIG)
    policy_results, scene_dist = evaluate_policies(file_records, scene_cfg)

    # Print scene distribution and per-policy reports
    print_scene_dist(scene_dist)
    for pname, r in policy_results.items():
        print_policy_report(pname, r, n_total)
    print_comparison_table(policy_results)

    # Save JSON
    out_dir = args.out or os.path.dirname(os.path.abspath(args.baseline))
    os.makedirs(out_dir, exist_ok=True)
    ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"policy_report_{ts}.json")

    with open(out_path, "w") as fh:
        json.dump(
            {
                "meta": {
                    "timestamp":        ts,
                    "baseline_file":    args.baseline,
                    "n_files":          n_total,
                    "high_conf_thresh": HIGH_CONF,
                    "scene_config":     scene_cfg,
                    "strict_music_cfg": dict(STRICT_MUSIC_CFG),
                },
                "scene_distribution": scene_dist,
                "policies":           policy_results,
            },
            fh,
            indent=2,
        )
    print(f"Policy report saved: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
