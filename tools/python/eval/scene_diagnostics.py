#!/usr/bin/env python3
"""
scene_diagnostics.py
Diagnostic report: scene classifier behaviour broken down by true audio label.

Loads a gate_eval.py baseline JSON, classifies every chunk with the scene
classifier (mirrored from scene.cpp, with confidence scores), then reports:

  - Per-label scene distribution at the file level
  - clean_speech files the classifier assigns MUSIC (false MUSIC labels)
  - music files the classifier does NOT assign MUSIC (missed detections)
  - Feature distributions (flatness, band_low, band_mid, band_high, centroid_hz)
    for each misclassified group
  - Music classification confidence distribution by true label

Two configs are compared:
  default   — DEFAULT_SCENE_CONFIG (music_band_low_min=0.18, music_band_high_min=0.08)
  strict    — STRICT_MUSIC_CFG     (music_band_low_min=0.25, music_band_high_min=0.12)

Audio concepts explained:
  Why speech can look like music to a band-energy rule:
    A tonal (low flatness) male voice can have significant bass energy (band_low)
    from the fundamental frequency and low harmonics, plus treble energy (band_high)
    from fricatives and upper harmonics. If both exceed the MUSIC thresholds,
    the current rule fires. Raising music_band_low_min from 0.18 to 0.25 requires
    more bass energy than most voices can supply, reducing the false positive rate.

  Why confidence-based rejection is safer than hard rejection:
    A hard reject on scene=MUSIC discards any file where no chunk is classified
    as SPEECH/MIXED/UNKNOWN. When the MUSIC rule is too permissive, this silently
    drops real speech. A confidence threshold lets through borderline classifications
    (low confidence = features barely satisfy the rule) while still catching files
    where features strongly match the music pattern.

  Why scene and gate quality should be evaluated separately:
    The gate decides whether audio has enough energy and structure to be worth
    transcribing. The scene classifier decides what kind of audio it is. A gate
    false accept (music passes the gate) is a different failure mode from a scene
    mis-classification (speech looks like music). Mixing them in a single hard policy
    allows one weak component to create compounding errors.

Output:
  - Human-readable report to stdout
  - scene_diagnostics_TIMESTAMP.json to --out dir

Usage:
    python tools/python/eval/scene_diagnostics.py baseline.json
    python tools/python/eval/scene_diagnostics.py baseline.json \\
        --out benchmarks/results/scene_diagnostics/
"""

import argparse
import datetime
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import gate_scene_policy as gsp  # noqa: E402

# Features to include in per-group summaries.
DISPLAY_FEATURES = ["flatness", "band_low", "band_mid", "band_high", "centroid_hz"]


def _feature_summary(chunks: List[Dict]) -> Dict:
    """Percentile stats for each display feature across a list of chunk dicts.

    Returns {} when chunks is empty.
    Audio use: identifies which feature(s) drive mis-classification by showing
    how the distributions of flatness/band_* differ between correct and incorrect groups.
    """
    if not chunks:
        return {}
    result = {}
    for key in DISPLAY_FEATURES:
        vals = [float(c.get(key, 0.0)) for c in chunks]
        arr = np.array(vals, dtype=np.float64)
        p25, p50, p75 = np.percentile(arr, [25.0, 50.0, 75.0])
        result[key] = {
            "mean": float(arr.mean()),
            "p25":  float(p25),
            "p50":  float(p50),
            "p75":  float(p75),
            "min":  float(arr.min()),
            "max":  float(arr.max()),
        }
    return result


def run_diagnostics(
    file_records: List[Dict],
    scene_cfg: Dict,
    strict_cfg: Dict,
) -> Dict:
    """Full diagnostic pass over all file records.

    Returns a structured dict with per-label distributions, misclassification
    lists, feature summaries, and confidence distributions.
    """
    label_scene_dist_default: Dict[str, Dict[str, int]] = {}
    label_scene_dist_strict:  Dict[str, Dict[str, int]] = {}

    # Files: clean_speech classified as MUSIC at the file level
    speech_as_music_default: List[Dict] = []
    speech_as_music_strict:  List[Dict] = []

    # Files: music-label NOT classified as MUSIC at the file level
    music_missed_default: List[Dict] = []

    # Chunk pools for feature analysis
    # clean_speech chunks that got MUSIC label under default config
    speech_chunks_as_music: List[Dict] = []
    # music-label chunks that got MUSIC label (correctly classified)
    music_chunks_correct: List[Dict] = []
    # music-label chunks that did NOT get MUSIC label (missed)
    music_chunks_missed: List[Dict] = []

    # Confidence of MUSIC classifications, grouped by true label
    music_conf_by_label: Dict[str, List[float]] = {}

    for r in file_records:
        true_label = r.get("label", "")
        path       = r.get("path", "")
        chunks     = r.get("chunks", [])

        # --- Default config ---
        def_results  = [gsp.classify_chunk_with_conf(c, scene_cfg) for c in chunks]
        def_labels   = [x[0] for x in def_results]
        def_confs    = [x[1] for x in def_results]
        def_scene    = gsp.file_scene_label(def_labels)

        # --- Strict music config ---
        str_results  = [gsp.classify_chunk_with_conf(c, strict_cfg) for c in chunks]
        str_labels   = [x[0] for x in str_results]
        str_scene    = gsp.file_scene_label(str_labels)

        # Per-label scene distribution (file-level)
        label_scene_dist_default.setdefault(true_label, {})
        label_scene_dist_default[true_label][def_scene] = (
            label_scene_dist_default[true_label].get(def_scene, 0) + 1
        )

        label_scene_dist_strict.setdefault(true_label, {})
        label_scene_dist_strict[true_label][str_scene] = (
            label_scene_dist_strict[true_label].get(str_scene, 0) + 1
        )

        # Confidence of MUSIC chunks, by true label
        for lbl, conf in zip(def_labels, def_confs):
            if lbl == gsp.MUSIC:
                music_conf_by_label.setdefault(true_label, []).append(conf)

        # --- Misclassification: clean_speech → MUSIC ---
        if true_label == "clean_speech":
            if def_scene == gsp.MUSIC:
                music_chunks_here = [
                    c for c, l in zip(chunks, def_labels) if l == gsp.MUSIC
                ]
                max_mc = max(
                    (conf for l, conf in zip(def_labels, def_confs) if l == gsp.MUSIC),
                    default=0.0,
                )
                speech_as_music_default.append({
                    "path":             os.path.basename(path),
                    "def_scene":        def_scene,
                    "def_chunk_labels": def_labels,
                    "n_chunks":         len(chunks),
                    "n_music_chunks":   sum(1 for l in def_labels if l == gsp.MUSIC),
                    "max_music_conf":   round(max_mc, 3),
                })
                speech_chunks_as_music.extend(music_chunks_here)

            if str_scene == gsp.MUSIC:
                speech_as_music_strict.append({
                    "path":             os.path.basename(path),
                    "str_scene":        str_scene,
                    "str_chunk_labels": str_labels,
                })

        # --- Misclassification: music → non-MUSIC ---
        if true_label == "music":
            if def_scene == gsp.MUSIC:
                # Correctly detected — collect MUSIC chunks for feature summary
                music_chunks_correct.extend(
                    c for c, l in zip(chunks, def_labels) if l == gsp.MUSIC
                )
            else:
                # Missed detection
                music_missed_default.append({
                    "path":             os.path.basename(path),
                    "def_scene":        def_scene,
                    "def_chunk_labels": def_labels,
                })
                music_chunks_missed.extend(chunks)

    return {
        "label_scene_distribution": {
            "default_config": label_scene_dist_default,
            "strict_config":  label_scene_dist_strict,
        },
        "misclassification": {
            "speech_as_music_default":    speech_as_music_default,
            "speech_as_music_strict":     speech_as_music_strict,
            "music_missed_default":       music_missed_default,
        },
        "counts": {
            "speech_total":           sum(1 for r in file_records if r.get("label") == "clean_speech"),
            "music_total":            sum(1 for r in file_records if r.get("label") == "music"),
            "speech_as_music_default": len(speech_as_music_default),
            "speech_as_music_strict":  len(speech_as_music_strict),
            "music_missed_default":    len(music_missed_default),
            "music_correct_default":   sum(1 for r in file_records if r.get("label") == "music")
                                       - len(music_missed_default),
        },
        "feature_summary": {
            "speech_chunks_classified_as_music_default":
                _feature_summary(speech_chunks_as_music),
            "music_chunks_correctly_classified":
                _feature_summary(music_chunks_correct),
            "music_chunks_missed_by_classifier":
                _feature_summary(music_chunks_missed),
        },
        "music_conf_by_label": {
            k: sorted(v) for k, v in music_conf_by_label.items()
        },
        "configs": {
            "default": scene_cfg,
            "strict":  strict_cfg,
        },
    }


# ---------------------------------------------------------------------------
# Print functions
# ---------------------------------------------------------------------------

def print_label_dist(label_scene_dist: Dict, config_name: str) -> None:
    all_scenes = [gsp.SPEECH, gsp.MIXED, gsp.MUSIC, gsp.NOISE, gsp.SILENCE, gsp.UNKNOWN]
    col_w = 8
    hdr = "  ".join(f"{s:>{col_w}}" for s in all_scenes)
    print()
    print(f"File-level scene distribution  [{config_name}]")
    print(f"  {'Label':35s}  {hdr}  {'Total':>6}")
    print("  " + "-" * 92)
    for true_label in sorted(label_scene_dist):
        dist  = label_scene_dist[true_label]
        total = sum(dist.values())
        row   = "  ".join(f"{dist.get(s, 0):>{col_w}d}" for s in all_scenes)
        print(f"  {true_label:35s}  {row}  {total:>6d}")


def print_misclassification(diag: Dict) -> None:
    counts = diag["counts"]
    mc     = diag["misclassification"]

    print()
    print("=" * 70)
    print("SCENE MISCLASSIFICATION ANALYSIS")
    print("=" * 70)

    # clean_speech → MUSIC
    n_sam = counts["speech_as_music_default"]
    n_st  = counts["speech_total"]
    n_ss  = counts["speech_as_music_strict"]
    print()
    print(f"clean_speech files classified as MUSIC at file level:")
    print(f"  default config : {n_sam:3d} / {n_st}  ({n_sam/max(n_st,1):.0%})")
    print(f"  strict  config : {n_ss:3d} / {n_st}  ({n_ss/max(n_st,1):.0%})")
    print(f"  improvement    : {n_sam - n_ss} fewer false MUSIC labels with strict thresholds")
    if mc["speech_as_music_default"]:
        print()
        print("  Files (default config):")
        for e in mc["speech_as_music_default"][:20]:
            nm = e["n_music_chunks"]
            nc = e["n_chunks"]
            mc_ = e["max_music_conf"]
            print(
                f"    {e['path']:50s}  "
                f"MUSIC_chunks={nm}/{nc}  max_conf={mc_:.2f}"
            )
        n_extra = len(mc["speech_as_music_default"]) - 20
        if n_extra > 0:
            print(f"    ... ({n_extra} more)")

    # music → non-MUSIC
    n_mm = counts["music_missed_default"]
    n_mt = counts["music_total"]
    n_mc = counts["music_correct_default"]
    print()
    print(f"music files NOT classified as MUSIC (missed detections, default config):")
    print(f"  missed  : {n_mm:3d} / {n_mt}  ({n_mm/max(n_mt,1):.0%})")
    print(f"  correct : {n_mc:3d} / {n_mt}  ({n_mc/max(n_mt,1):.0%})")
    if mc["music_missed_default"]:
        print()
        print("  Files (default config):")
        for e in mc["music_missed_default"][:20]:
            cl = " ".join(e["def_chunk_labels"])
            print(f"    {e['path']:50s}  scene={e['def_scene']}  chunks=[{cl}]")
        n_extra = len(mc["music_missed_default"]) - 20
        if n_extra > 0:
            print(f"    ... ({n_extra} more)")


def print_feature_summary(diag: Dict) -> None:
    feat_groups = diag.get("feature_summary", {})
    if not feat_groups:
        return

    group_labels = {
        "speech_chunks_classified_as_music_default":
            "clean_speech chunks classified as MUSIC (default cfg)",
        "music_chunks_correctly_classified":
            "music chunks correctly classified as MUSIC",
        "music_chunks_missed_by_classifier":
            "music chunks NOT classified as MUSIC (missed)",
    }

    print()
    print("=" * 70)
    print("FEATURE DISTRIBUTION BY GROUP")
    print("  p50=median, p25/p75=interquartile range")
    print("=" * 70)

    for group_key, group_label in group_labels.items():
        summary = feat_groups.get(group_key, {})
        if not summary:
            print(f"\n  {group_label}  (no chunks)")
            continue
        print(f"\n  {group_label}")
        print(f"  {'Feature':20s}  {'p25':>8}  {'p50':>8}  {'p75':>8}  {'mean':>8}")
        print("  " + "-" * 58)
        for fk in DISPLAY_FEATURES:
            if fk not in summary:
                continue
            s = summary[fk]
            print(
                f"  {fk:20s}  {s['p25']:>8.4f}  {s['p50']:>8.4f}  "
                f"{s['p75']:>8.4f}  {s['mean']:>8.4f}"
            )

    # Data-driven analysis of which strict threshold drives the improvement.
    # Strict config: music_band_low_min=0.25, music_band_high_min=0.12
    sm = feat_groups.get("speech_chunks_classified_as_music_default", {})
    _strict = gsp.STRICT_MUSIC_CFG
    _default = gsp.DEFAULT_SCENE_CONFIG
    strict_low  = _strict.get("music_band_low_min",  0.25)
    strict_high = _strict.get("music_band_high_min", 0.12)
    if sm.get("band_low") and sm.get("band_high"):
        bl_speech_p75 = sm["band_low"]["p75"]
        bh_speech_p25 = sm["band_high"]["p25"]
        bh_speech_p50 = sm["band_high"]["p50"]
        print()
        print("  Strict threshold analysis:")
        print(
            f"    band_low  default={_default['music_band_low_min']:.2f}  "
            f"strict={strict_low:.2f}  speech-as-music p75={bl_speech_p75:.3f}"
        )
        if bl_speech_p75 < strict_low:
            print("      => strict band_low RESOLVES overlap (speech p75 below strict threshold)")
        else:
            print(
                "      => strict band_low does NOT resolve overlap "
                f"(speech band_low is high throughout; p75={bl_speech_p75:.3f} >> {strict_low:.2f})"
            )
        print(
            f"    band_high default={_default['music_band_high_min']:.2f}  "
            f"strict={strict_high:.2f}  speech-as-music p25={bh_speech_p25:.3f}  p50={bh_speech_p50:.3f}"
        )
        if bh_speech_p25 < strict_high:
            print(
                f"      => strict band_high IS the effective discriminator: "
                f"speech files with band_high < {strict_high:.2f} no longer trigger MUSIC rule"
            )
        else:
            print("      => strict band_high does NOT resolve overlap")


def print_conf_summary(diag: Dict) -> None:
    conf_by_label = diag.get("music_conf_by_label", {})
    if not conf_by_label:
        return

    print()
    print("=" * 70)
    print("MUSIC CLASSIFICATION CONFIDENCE  (chunks classified as MUSIC, default config)")
    print("=" * 70)
    print(f"  {'Label':35s}  {'N':>5}  {'p25':>6}  {'p50':>6}  {'p75':>6}  {'mean':>6}")
    print("  " + "-" * 68)
    for label in sorted(conf_by_label):
        confs = conf_by_label[label]
        if not confs:
            continue
        arr = np.array(confs)
        p25, p50, p75 = np.percentile(arr, [25.0, 50.0, 75.0])
        print(
            f"  {label:35s}  {len(confs):>5d}  "
            f"{p25:>6.3f}  {p50:>6.3f}  {p75:>6.3f}  {arr.mean():>6.3f}"
        )
    print()
    print(
        f"  Note: HIGH_CONF threshold in policy B/C = {gsp.HIGH_CONF:.2f}. "
        "Chunks above this are\n"
        "  considered high-confidence music and will trigger a policy skip."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("baseline", help="Baseline JSON from gate_eval.py")
    ap.add_argument(
        "--out",
        default="benchmarks/results/scene_diagnostics",
        help="Output directory for diagnostics JSON",
    )
    args = ap.parse_args()

    with open(args.baseline) as fh:
        artifact = json.load(fh)

    file_records: List[Dict] = artifact["files"]
    print(f"Loaded {len(file_records)} file records.", file=sys.stderr)

    scene_cfg  = dict(gsp.DEFAULT_SCENE_CONFIG)
    strict_cfg = dict(gsp.STRICT_MUSIC_CFG)
    diag = run_diagnostics(file_records, scene_cfg, strict_cfg)

    print_label_dist(diag["label_scene_distribution"]["default_config"], "default config")
    print_label_dist(diag["label_scene_distribution"]["strict_config"],  "strict config")
    print_misclassification(diag)
    print_feature_summary(diag)
    print_conf_summary(diag)

    # Save JSON (numpy scalars need conversion)
    def _to_python(obj):
        if isinstance(obj, dict):
            return {k: _to_python(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_python(x) for x in obj]
        if hasattr(obj, "item"):  # numpy scalar
            return obj.item()
        return obj

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"scene_diagnostics_{ts}.json")

    with open(out_path, "w") as fh:
        json.dump(_to_python(diag), fh, indent=2)

    print(f"\nDiagnostics saved: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
