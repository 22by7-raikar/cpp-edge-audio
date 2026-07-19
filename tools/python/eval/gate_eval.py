#!/usr/bin/env python3
"""
gate_eval.py
Compute gate metrics for every file in a JSONL label file, simulate gate
decisions using the current GateConfig defaults, and produce a baseline report.

The gate logic is a Python mirror of:
  runtime/cpp/src/gate/gate.cpp     (decision logic)
  runtime/cpp/src/gate/features.cpp (FFT-based feature extraction)

Feature computation uses numpy.fft and matches the C++ Cooley-Tukey radix-2 DIT
implementation to within floating-point rounding error. Field names follow the
JSON schema defined in .github/instructions/schema.instructions.md.

File-level decision from chunks:
  any PASS              -> PASS
  no PASS + any BORDER  -> BORDERLINE
  all FAIL              -> FAIL

Usage:
    python tools/python/eval/gate_eval.py
    python tools/python/eval/gate_eval.py --labels data/labels/eval_subset.jsonl
    python tools/python/eval/gate_eval.py --out benchmarks/results/gate_calibration/
    python tools/python/eval/gate_eval.py --chunk-sec 5.0
"""

import argparse
import datetime
import json
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

# Repo root: tools/python/eval/ -> tools/python/ -> tools/ -> repo root
_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
)


def resolve_path(path: str) -> str:
    """Resolve a label/manifest path to absolute.

    Repo-relative paths (no leading '/') are joined with the repo root.
    Absolute paths pass through unchanged.
    Supports JSONL files written both before (absolute) and after
    (repo-relative) the portability change.
    """
    if os.path.isabs(path):
        return path
    return os.path.join(_REPO_ROOT, path)


# ---------------------------------------------------------------------------
# Default GateConfig — exact mirror of runtime/cpp/src/gate/gate.h defaults.
# Change these only when gate.h defaults change.
# ---------------------------------------------------------------------------
DEFAULT_CONFIG: Dict = {
    # Time-domain thresholds
    "rms_borderline_min":    0.001,
    "rms_min":               0.003,
    "max_silence_ratio":     0.90,
    "max_clipping_ratio":    0.05,
    "silence_thresh":        0.005,
    "clipping_thresh":       0.99,
    "zcr_max_noise":         400.0,
    # Spectral
    "min_active_frame_frac": 0.10,
    "spectral_flatness_max": 0.90,
    "spectral_flatness_warn":0.72,
    "min_band_mid":          0.10,
    "max_band_high":         0.70,
    # FrameConfig (passed to extract_chunk_features in C++)
    "frame_size":            512,
    "hop_size":              256,
    "rolloff_percentile":    0.85,
    "active_rms_thresh":     0.005,
    "band_low_max_hz":       500.0,
    "band_mid_max_hz":       4000.0,
}

_EPS = 1e-10


# ---------------------------------------------------------------------------
# Feature extraction — mirrors features.cpp
# ---------------------------------------------------------------------------

def _hann(n: int) -> np.ndarray:
    """Symmetric Hann window: w[i] = 0.5*(1-cos(2*pi*i/(n-1))).
    Matches C++ make_hann() and numpy's np.hanning(n)."""
    return np.hanning(n).astype(np.float32)


def _mag_spectrum(frame: np.ndarray, hann: np.ndarray) -> np.ndarray:
    """One-sided magnitude spectrum (n//2+1 bins) of a Hann-windowed frame.
    Matches C++ magnitude_spectrum() using the same DFT definition."""
    return np.abs(np.fft.rfft(frame * hann)).astype(np.float32)


def _compute_frame_features(
    mag: np.ndarray,
    prev_mag: Optional[np.ndarray],
    time_samples: np.ndarray,
    sample_rate: int,
    cfg: Dict,
) -> Dict:
    """Per-frame spectral features — mirrors compute_frame_features() in features.cpp.

    Audio concept: each 512-sample (32 ms) frame is windowed with Hann to reduce
    spectral leakage, then FFT-transformed. The power spectrum is the squared
    magnitude. Flatness near 1 means broadband noise; near 0 means tonal/voiced
    content. Centroid is the energy-weighted average frequency. Rolloff is the
    frequency below which 85% of spectral energy sits.
    """
    frame_size = cfg["frame_size"]
    bin_hz = sample_rate / frame_size
    n_bins = len(mag)

    # Frame RMS (time-domain)
    frame_rms = float(np.sqrt(np.mean(time_samples.astype(np.float64) ** 2)))

    power = mag.astype(np.float64) ** 2
    total_power = float(power.sum())

    if total_power < _EPS:
        # Silent frame — all spectral features are zero, same as C++ early return
        return {
            "frame_rms":   frame_rms,
            "flatness":    0.0,
            "centroid_hz": 0.0,
            "rolloff_hz":  0.0,
            "flux":        0.0,
            "band_low":    0.0,
            "band_mid":    0.0,
            "band_high":   0.0,
        }

    # Spectral flatness = geometric_mean(power) / arithmetic_mean(power).
    # Computed in log space to avoid underflow, clamped to [0,1].
    log_sum = float(np.sum(np.log(power + _EPS)))
    geom_mean = math.exp(log_sum / n_bins)
    arith_mean = total_power / n_bins
    flatness = min(geom_mean / (arith_mean + _EPS), 1.0)

    # Spectral centroid: energy-weighted bin index converted to Hz.
    k_arr = np.arange(n_bins, dtype=np.float64)
    centroid_hz = float(np.dot(k_arr, power) / (total_power + _EPS)) * bin_hz

    # Spectral rolloff: lowest bin where cumulative power >= rolloff_percentile.
    rolloff_threshold = cfg["rolloff_percentile"] * total_power
    cumsum = np.cumsum(power)
    above = np.where(cumsum >= rolloff_threshold)[0]
    rolloff_hz = (float(above[0]) * bin_hz) if len(above) else float(n_bins - 1) * bin_hz

    # Spectral flux: sum of squared magnitude differences vs previous frame.
    # Audio concept: flux measures temporal change in the spectrum; voiced speech
    # has high flux (changing formants), stationary noise has low flux.
    flux = 0.0
    if prev_mag is not None:
        diff = mag - prev_mag
        flux = float(np.dot(diff, diff))

    # Band energy fractions: proportion of power in each frequency region.
    # low=[0,500Hz], mid=[500,4000Hz], high=[4000,Nyquist].
    low_bin = min(int(cfg["band_low_max_hz"] / bin_hz), n_bins - 1)
    mid_bin = min(int(cfg["band_mid_max_hz"] / bin_hz), n_bins - 1)
    e_low  = float(power[: low_bin + 1].sum())
    e_mid  = float(power[low_bin + 1: mid_bin + 1].sum())
    e_high = float(power[mid_bin + 1:].sum())
    denom  = total_power + _EPS

    return {
        "frame_rms":   frame_rms,
        "flatness":    flatness,
        "centroid_hz": centroid_hz,
        "rolloff_hz":  rolloff_hz,
        "flux":        flux,
        "band_low":    e_low  / denom,
        "band_mid":    e_mid  / denom,
        "band_high":   e_high / denom,
    }


def extract_chunk_features(
    samples: np.ndarray,
    sample_rate: int,
    cfg: Dict,
) -> Dict:
    """Chunk-level feature aggregation — mirrors extract_chunk_features() in features.cpp.

    Runs a sliding window of frame_size=512 samples, hop=256 (50% overlap) over
    the chunk, computes per-frame features, then averages them. Returns the means
    and the active frame fraction (frames where frame_rms >= active_rms_thresh).
    """
    frame_size = cfg["frame_size"]
    hop_size   = cfg["hop_size"]
    hann       = _hann(frame_size)

    frames: List[Dict] = []
    prev_mag: Optional[np.ndarray] = None
    n = len(samples)

    for offset in range(0, n - frame_size + 1, hop_size):
        frame = samples[offset: offset + frame_size].astype(np.float32)
        mag   = _mag_spectrum(frame, hann)
        ff    = _compute_frame_features(mag, prev_mag, frame, sample_rate, cfg)
        frames.append(ff)
        prev_mag = mag

    if not frames:
        return {
            "flatness": 0.0, "centroid_hz": 0.0, "rolloff_hz": 0.0,
            "flux": 0.0, "band_low": 0.0, "band_mid": 0.0, "band_high": 0.0,
            "active_frac": 0.0, "n_frames": 0,
        }

    n_frames = len(frames)
    active   = sum(1 for f in frames if f["frame_rms"] >= cfg["active_rms_thresh"])

    def _mean(key: str) -> float:
        return sum(f[key] for f in frames) / n_frames

    return {
        "flatness":    _mean("flatness"),
        "centroid_hz": _mean("centroid_hz"),
        "rolloff_hz":  _mean("rolloff_hz"),
        "flux":        _mean("flux"),
        "band_low":    _mean("band_low"),
        "band_mid":    _mean("band_mid"),
        "band_high":   _mean("band_high"),
        "active_frac": active / n_frames,
        "n_frames":    n_frames,
    }


# ---------------------------------------------------------------------------
# Time-domain metrics + full chunk metrics — mirrors compute_metrics() in gate.cpp
# ---------------------------------------------------------------------------

def compute_chunk_metrics(
    samples: np.ndarray,
    sample_rate: int,
    cfg: Dict,
) -> Dict:
    """Time-domain and spectral metrics for one chunk.

    Time-domain concepts:
      RMS: root-mean-square energy — proxy for perceived loudness.
      silence_ratio: fraction of samples below silence_thresh (0.005). High
        values mean the chunk is mostly quiet, not necessarily silent.
      clipping_ratio: fraction of samples at or above clipping_thresh (0.99).
        ADC clipping produces flat-tops in the waveform; even small amounts
        degrade ASR quality.
      ZCR (zero-crossing rate): number of sign changes per second. Voiced speech
        is typically 60-200/s; unvoiced consonants ~200-400/s; white noise >400/s.
        Used as a secondary noise discriminator alongside spectral flatness.
    """
    n = len(samples)
    if n == 0:
        return {}

    duration_sec = n / sample_rate

    # RMS
    rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))

    # Silence and clipping ratios
    abs_s          = np.abs(samples)
    silence_ratio  = float(np.sum(abs_s < cfg["silence_thresh"]) / n)
    clipping_ratio = float(np.sum(abs_s >= cfg["clipping_thresh"]) / n)

    # ZCR: count transitions between (>=0) and (<0), matching C++ semantics.
    pos = (samples >= 0.0)
    zcr = float(np.sum(pos[:-1] != pos[1:])) / duration_sec if duration_sec > 0 else 0.0

    spec = extract_chunk_features(samples, sample_rate, cfg)

    return {
        "duration_sec":  duration_sec,
        "rms":           rms,
        "silence_ratio": silence_ratio,
        "clipping_ratio":clipping_ratio,
        "zcr":           zcr,
        # Spectral — use schema-compliant field names (schema.instructions.md)
        "flatness":      spec["flatness"],
        "centroid_hz":   spec["centroid_hz"],
        "rolloff_hz":    spec["rolloff_hz"],
        "flux":          spec["flux"],
        "band_low":      spec["band_low"],
        "band_mid":      spec["band_mid"],
        "band_high":     spec["band_high"],
        "active_frac":   spec["active_frac"],
    }


# ---------------------------------------------------------------------------
# Gate decision — mirrors evaluate_chunk() in gate.cpp, steps 1-10
# ---------------------------------------------------------------------------

def evaluate_chunk(metrics: Dict, cfg: Dict) -> Tuple[str, str]:
    """Return (decision, reason). Reason strings are stable; do not rename.

    Decision order mirrors gate.cpp exactly:
      1. RMS hard floor (rms_too_low)
      2. Silence ratio  (high_silence_ratio)
      3. Clipping       (high_clipping_ratio)
      4. Active frames  (low_active_frame_fraction)
      5. Flatness FAIL  (stationary_noise_like)
      6. Flatness+ZCR   (stationary_noise_like)
      7. Band-mid       (weak_mid_band_speech_presence)
      8. Band-high      (excessive_high_band_energy)
      9. RMS borderline (borderline_low_energy)
     10. Flatness warn  (borderline_noisy_speech)
     11. PASS           (ok)

    Audio concept: the gate is a sequential filter that checks cheapest features
    first (energy) before computing FFT-based spectral features. This mirrors
    the C++ evaluation order and lets callers short-circuit on the reason string.
    """
    m = metrics

    if m["rms"] < cfg["rms_borderline_min"]:
        return "FAIL", "rms_too_low"
    if m["silence_ratio"] > cfg["max_silence_ratio"]:
        return "FAIL", "high_silence_ratio"
    if m["clipping_ratio"] > cfg["max_clipping_ratio"]:
        return "FAIL", "high_clipping_ratio"
    if m["active_frac"] < cfg["min_active_frame_frac"]:
        return "FAIL", "low_active_frame_fraction"
    if m["flatness"] > cfg["spectral_flatness_max"]:
        return "FAIL", "stationary_noise_like"
    if m["flatness"] > cfg["spectral_flatness_warn"] and m["zcr"] > cfg["zcr_max_noise"]:
        return "FAIL", "stationary_noise_like"
    if m["band_mid"] < cfg["min_band_mid"]:
        return "FAIL", "weak_mid_band_speech_presence"
    if m["band_high"] > cfg["max_band_high"]:
        return "FAIL", "excessive_high_band_energy"
    if m["rms"] < cfg["rms_min"]:
        return "BORDERLINE", "borderline_low_energy"
    if m["flatness"] > cfg["spectral_flatness_warn"]:
        return "BORDERLINE", "borderline_noisy_speech"
    return "PASS", "ok"


def _file_decision(chunk_decisions: List[str]) -> str:
    """Aggregate chunk-level decisions to a single file-level decision.

    Logic: the gate's job is to decide whether any chunk in the file is worth
    transcribing. If any chunk PASSes, the file should be transcribed. If none
    pass but some are BORDERLINE, the file is uncertain. All-FAIL means reject.
    """
    if "PASS" in chunk_decisions:
        return "PASS"
    if "BORDERLINE" in chunk_decisions:
        return "BORDERLINE"
    return "FAIL"


# ---------------------------------------------------------------------------
# Audio loading and chunking
# ---------------------------------------------------------------------------

def load_audio(path: str) -> Tuple[np.ndarray, int]:
    """Load mono float32 audio using soundfile (supports WAV, FLAC, OGG, etc.).
    Multi-channel audio is mixed to mono by averaging channels."""
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    return mono, sr


def chunk_audio(samples: np.ndarray, sample_rate: int, chunk_sec: float) -> List[np.ndarray]:
    """Split samples into non-overlapping chunks of chunk_sec seconds.
    The final short chunk is kept if non-empty (gate handles short chunks).
    """
    chunk_len = int(chunk_sec * sample_rate)
    if chunk_len <= 0:
        return [samples]
    return [
        samples[start: start + chunk_len]
        for start in range(0, len(samples), chunk_len)
        if start < len(samples)
    ]


# ---------------------------------------------------------------------------
# Per-file evaluation
# ---------------------------------------------------------------------------

def evaluate_file(path: str, cfg: Dict, chunk_sec: float) -> Dict:
    """Load audio, chunk, compute metrics, apply gate. Returns file record dict."""
    try:
        samples, sr = load_audio(path)
    except Exception as exc:
        return {"error": str(exc), "chunks": [], "file_decision": "ERROR"}

    chunks = chunk_audio(samples, sr, chunk_sec)
    chunk_records: List[Dict] = []
    chunk_decisions: List[str] = []

    for i, chunk_samples in enumerate(chunks):
        start_sec = i * chunk_sec
        end_sec   = start_sec + len(chunk_samples) / sr
        metrics   = compute_chunk_metrics(chunk_samples, sr, cfg)
        decision, reason = evaluate_chunk(metrics, cfg)
        chunk_decisions.append(decision)
        chunk_records.append({
            "idx":       i,
            "start_sec": round(start_sec, 3),
            "end_sec":   round(end_sec, 3),
            "decision":  decision,
            "reason":    reason,
            **{k: round(v, 6) if isinstance(v, float) else v for k, v in metrics.items()},
        })

    return {
        "chunks":        chunk_records,
        "file_decision": _file_decision(chunk_decisions),
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def build_report(file_records: List[Dict]) -> Dict:
    """Compute all baseline metrics from file records.

    Terms:
      False accept (FA): gate PASSes a file that should NOT be transcribed.
        These waste ASR compute and may inject hallucinations into the output.
      False reject (FR): gate FAILs a file that SHOULD be transcribed.
        These cause missed transcriptions — content is silently dropped.
      Borderline: gate is uncertain. In deployment, these go to ASR anyway;
        borderline decisions are calibration feedback, not deployment failures.

    Calibration goal: before training any learned model, find rule-based
    thresholds that minimise FA+FR on real data. This gives a reliable baseline
    and exposes which audio categories the current rules mis-classify.
    """
    non_error = [r for r in file_records if r["file_decision"] != "ERROR"]
    n_total   = len(non_error)

    counts: Dict[str, int] = {"PASS": 0, "BORDERLINE": 0, "FAIL": 0}
    for r in non_error:
        counts[r["file_decision"]] += 1

    # By label
    by_label: Dict[str, Dict] = {}
    for r in non_error:
        lbl = r["label"]
        if lbl not in by_label:
            by_label[lbl] = {"PASS": 0, "BORDERLINE": 0, "FAIL": 0, "total": 0}
        by_label[lbl][r["file_decision"]] += 1
        by_label[lbl]["total"] += 1

    # By should_transcribe
    by_st: Dict[str, Dict] = {
        "yes": {"PASS": 0, "BORDERLINE": 0, "FAIL": 0, "total": 0},
        "no":  {"PASS": 0, "BORDERLINE": 0, "FAIL": 0, "total": 0},
    }
    for r in non_error:
        st = r.get("should_transcribe", "")
        if st in by_st:
            by_st[st][r["file_decision"]] += 1
            by_st[st]["total"] += 1

    n_should_yes = by_st["yes"]["total"]
    n_should_no  = by_st["no"]["total"]

    # False accepts / rejects (file-level)
    false_accepts = [
        r["path"] for r in non_error
        if r.get("should_transcribe") == "no" and r["file_decision"] == "PASS"
    ]
    false_rejects = [
        r["path"] for r in non_error
        if r.get("should_transcribe") == "yes" and r["file_decision"] == "FAIL"
    ]

    # Borderline by label
    borderline_by_label: Dict[str, int] = {}
    for r in non_error:
        if r["file_decision"] == "BORDERLINE":
            lbl = r["label"]
            borderline_by_label[lbl] = borderline_by_label.get(lbl, 0) + 1

    # FAIL reason distribution (chunk-level, all chunks in FAILed files)
    fail_reasons: Dict[str, int] = {}
    for r in non_error:
        if r["file_decision"] == "FAIL":
            for c in r.get("chunks", []):
                reason = c.get("reason", "")
                if reason:
                    fail_reasons[reason] = fail_reasons.get(reason, 0) + 1

    far = len(false_accepts) / max(n_should_no, 1)
    frr = len(false_rejects) / max(n_should_yes, 1)

    return {
        "n_total":              n_total,
        "counts":               counts,
        "pass_rate":            counts["PASS"] / max(n_total, 1),
        "borderline_rate":      counts["BORDERLINE"] / max(n_total, 1),
        "fail_rate":            counts["FAIL"] / max(n_total, 1),
        "by_label":             by_label,
        "by_should_transcribe": by_st,
        "false_accepts":        false_accepts,
        "false_rejects":        false_rejects,
        "false_accept_rate":    far,
        "false_reject_rate":    frr,
        "borderline_by_label":  borderline_by_label,
        "fail_reason_counts":   fail_reasons,
        "n_errors":             sum(1 for r in file_records if r["file_decision"] == "ERROR"),
    }


def print_report(report: Dict) -> None:
    sep = "=" * 68
    print(f"\n{sep}")
    print("GATE BASELINE REPORT")
    print(sep)
    n = max(report["n_total"], 1)
    c = report["counts"]
    print(f"Total files : {report['n_total']}")
    print(f"  PASS       : {c['PASS']:4d}  ({c['PASS']/n:6.1%})")
    print(f"  BORDERLINE : {c['BORDERLINE']:4d}  ({c['BORDERLINE']/n:6.1%})")
    print(f"  FAIL       : {c['FAIL']:4d}  ({c['FAIL']/n:6.1%})")
    if report["n_errors"]:
        print(f"  ERRORS     : {report['n_errors']:4d}")

    print()
    print("By should_transcribe:")
    for st, vals in report["by_should_transcribe"].items():
        t = max(vals["total"], 1)
        print(
            f"  {st:3s}  total={vals['total']:4d}  "
            f"PASS={vals['PASS']:4d} ({vals['PASS']/t:5.1%})  "
            f"BORDERLINE={vals['BORDERLINE']:4d} ({vals['BORDERLINE']/t:5.1%})  "
            f"FAIL={vals['FAIL']:4d} ({vals['FAIL']/t:5.1%})"
        )

    print()
    fa  = report["false_accepts"]
    fr  = report["false_rejects"]
    n_no  = report["by_should_transcribe"]["no"]["total"]
    n_yes = report["by_should_transcribe"]["yes"]["total"]
    print(f"False accept rate  (no ->PASS) : {len(fa):4d} / {n_no}  = {report['false_accept_rate']:.4f}")
    print(f"False reject rate  (yes->FAIL) : {len(fr):4d} / {n_yes}  = {report['false_reject_rate']:.4f}")

    print()
    print("By label:")
    for label, vals in sorted(report["by_label"].items()):
        t = max(vals["total"], 1)
        print(
            f"  {label:35s}  total={vals['total']:4d}  "
            f"PASS={vals['PASS']:3d} ({vals['PASS']/t:5.1%})  "
            f"BORDER={vals['BORDERLINE']:3d} ({vals['BORDERLINE']/t:5.1%})  "
            f"FAIL={vals['FAIL']:3d} ({vals['FAIL']/t:5.1%})"
        )

    if report["borderline_by_label"]:
        print()
        print("Borderline files by label:")
        for label, cnt in sorted(report["borderline_by_label"].items(), key=lambda x: -x[1]):
            print(f"  {label:35s}  {cnt:4d}")

    if report["fail_reason_counts"]:
        print()
        print("FAIL chunk reason distribution:")
        for reason, cnt in sorted(report["fail_reason_counts"].items(), key=lambda x: -x[1]):
            print(f"  {reason:42s}  {cnt:5d}")

    if fa:
        print()
        print(f"False accepts ({len(fa)} files, should_transcribe=no but PASS):")
        for p in fa[:12]:
            print(f"  {os.path.basename(p)}")
        if len(fa) > 12:
            print(f"  ... ({len(fa)-12} more)")

    if fr:
        print()
        print(f"False rejects ({len(fr)} files, should_transcribe=yes but FAIL):")
        for p in fr[:12]:
            print(f"  {os.path.basename(p)}")
        if len(fr) > 12:
            print(f"  ... ({len(fr)-12} more)")

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
        "--labels",
        default="data/labels/eval_subset.jsonl",
        help="JSONL label file (default: data/labels/eval_subset.jsonl)",
    )
    ap.add_argument(
        "--out",
        default="benchmarks/results/gate_calibration/",
        help="Output directory for baseline JSON artifact",
    )
    ap.add_argument(
        "--chunk-sec",
        type=float,
        default=5.0,
        help="Chunk duration in seconds (default: 5.0, matches C++ default chunk_ms=5000)",
    )
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    labels: List[Dict] = []
    with open(args.labels) as fh:
        for line in fh:
            line = line.strip()
            if line:
                labels.append(json.loads(line))

    print(f"Loaded {len(labels)} labels from {args.labels}", file=sys.stderr)

    cfg = dict(DEFAULT_CONFIG)
    file_records: List[Dict] = []

    for i, entry in enumerate(labels):
        path = resolve_path(entry["path"])
        print(
            f"\r[{i+1:4d}/{len(labels)}] {os.path.basename(path)[:55]:55s}",
            end="",
            file=sys.stderr,
            flush=True,
        )
        result = evaluate_file(path, cfg, args.chunk_sec)
        file_records.append({
            "path":             path,
            "label":            entry.get("label", ""),
            "should_transcribe":entry.get("should_transcribe", ""),
            "duration_sec":     entry.get("duration_sec", 0.0),
            **result,
        })

    print(file=sys.stderr)

    report = build_report(file_records)
    print_report(report)

    # Save calibration JSON — input for gate_calibrate.py
    ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.out, f"baseline_{ts}.json")

    artifact = {
        "meta": {
            "timestamp":   ts,
            "labels_file": args.labels,
            "chunk_sec":   args.chunk_sec,
            "n_labels":    len(labels),
            "config":      cfg,
        },
        "files":   file_records,
        "summary": report,
    }

    with open(out_path, "w") as fh:
        json.dump(artifact, fh, indent=2)

    print(f"Baseline artifact : {out_path}", file=sys.stderr)
    print(f"Next step         : python tools/python/tuning/gate_calibrate.py {out_path}")


if __name__ == "__main__":
    main()
