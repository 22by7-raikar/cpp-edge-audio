#!/usr/bin/env python3
"""
gate_parity.py
Validate Python gate feature extraction against the C++ runtime.

Runs the C++ binary in --gate-only mode (no ASR) on a small deterministic audio
subset, reads the per-chunk metrics from the JSON output, runs the Python gate
logic (gate_eval.py) on the same files, then compares each metric with
per-metric absolute tolerances.

This is a regression test for feature parity. It does NOT validate correctness
of the gate thresholds — only that the Python and C++ implementations produce
the same feature values within floating-point rounding bounds.

Output:
  - Parity report to stdout
  - JSON file to --out dir if specified

Usage:
    python tools/python/eval/gate_parity.py
    python tools/python/eval/gate_parity.py --n 3
    python tools/python/eval/gate_parity.py --binary runtime/cpp/build/audio_pipeline
    python tools/python/eval/gate_parity.py --out benchmarks/results/gate_calibration/

Why tolerances are not zero:
  C++ uses float32 for frame-level computations (FrameFeatures) and accumulates
  to double at the chunk level. Python uses float64 throughout. FFT bin
  ordering and log/exp accumulation order differ. Time-domain features (RMS,
  ratios) are very close; spectral features can diverge by small amounts due to
  these systematic differences. Decision mismatches are expected near thresholds.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional, Tuple

# Import Python gate implementation from the same package.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import gate_eval  # noqa: E402

DEFAULT_BINARY  = "runtime/cpp/build/audio_pipeline"
DEFAULT_LABELS  = "data/labels/eval_subset.jsonl"
N_PER_LABEL     = 5  # files per label across all 7 eval labels

# Per-metric absolute tolerances.
# These reflect systematic differences between float32 C++ and float64 Python.
TOLERANCES: Dict[str, float] = {
    "rms":           1e-4,
    "silence_ratio": 1e-4,
    "clipping_ratio":1e-4,
    "zcr":           2.0,    # per-second rate; sign comparisons on float32 samples
    "flatness":      2e-3,   # geometric mean via log/exp; float32 accumulation
    "centroid_hz":   10.0,   # Hz; weighted average of float32 power bins
    "rolloff_hz":    50.0,   # Hz; percentile threshold crossing — can jump one bin
    "flux":          5e-3,   # sum of squared diffs; float32 to double
    "band_low":      2e-3,
    "band_mid":      2e-3,
    "band_high":     2e-3,
    "active_frac":   2e-2,   # frame count; RMS threshold applied to float32
}

# C++ JSON chunk field -> Python metric key mapping.
# These must match logger.cpp write_json() field names.
FIELD_MAP = {
    "rms":           "rms",
    "silence_ratio": "silence_ratio",
    "clipping_ratio":"clipping_ratio",
    "zcr":           "zcr",
    "flatness":      "flatness",
    "centroid_hz":   "centroid_hz",
    "rolloff_hz":    "rolloff_hz",
    "flux":          "flux",
    "band_low":      "band_low",
    "band_mid":      "band_mid",
    "band_high":     "band_high",
    "active_frac":   "active_frac",
}


def select_files(labels_path: str, n_per_label: int) -> List[Dict]:
    """Select up to n_per_label files per label, WAV and FLAC (deterministic: first n sorted).

    FLAC files are included so that labels sourced from LibriSpeech (clean_speech)
    are covered. FLAC files are converted to a temporary WAV before C++ validation;
    Python evaluation runs on the original file.
    """
    import json as _json
    labels: List[Dict] = []
    with open(labels_path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                labels.append(_json.loads(line))

    by_label: Dict[str, List[Dict]] = {}
    for entry in labels:
        if not entry["path"].lower().endswith((".wav", ".flac")):
            continue
        lbl = entry["label"]
        by_label.setdefault(lbl, []).append(entry)

    selected: List[Dict] = []
    for lbl in sorted(by_label):
        subset = sorted(by_label[lbl], key=lambda x: x["path"])[:n_per_label]
        selected.extend(subset)

    return selected


def run_cpp_gate_only(
    binary: str,
    wav_path: str,
    chunk_ms: int = 5000,
) -> Optional[List[Dict]]:
    """Run C++ binary in --gate-only mode on a WAV file.
    Returns list of chunk dicts (from bench-json output), or None on failure."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        json_path = tmp.name

    try:
        result = subprocess.run(
            [
                binary,
                "--input",    wav_path,
                "--chunk-ms", str(chunk_ms),
                "--gate-only",
                "--bench-json", json_path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            print(
                f"  C++ error on {os.path.basename(wav_path)}: "
                f"{result.stderr.strip()[:120]}",
                file=sys.stderr,
            )
            return None

        with open(json_path) as fh:
            data = json.load(fh)
        return data.get("chunks", [])

    except subprocess.TimeoutExpired:
        print(f"  C++ timeout on {os.path.basename(wav_path)}", file=sys.stderr)
        return None
    except Exception as exc:
        print(f"  C++ exception on {os.path.basename(wav_path)}: {exc}", file=sys.stderr)
        return None
    finally:
        try:
            os.unlink(json_path)
        except OSError:
            pass


def run_python_gate(wav_path: str, chunk_sec: float = 5.0) -> List[Dict]:
    """Run Python gate on an audio file. Returns list of chunk metric dicts."""
    cfg = dict(gate_eval.DEFAULT_CONFIG)
    result = gate_eval.evaluate_file(wav_path, cfg, chunk_sec)
    return result.get("chunks", [])


def _flac_to_wav(src: str, dst: str) -> None:
    """Convert src (any soundfile-readable format) to a 16-bit PCM WAV at dst."""
    import soundfile as sf
    data, sr = sf.read(src, dtype="int16", always_2d=False)
    sf.write(dst, data, sr, subtype="PCM_16")


def compare_chunks(
    cpp_chunks: List[Dict],
    py_chunks: List[Dict],
) -> Tuple[int, int, List[Dict]]:
    """Compare C++ and Python chunk records. Returns (n_ok, n_mismatch, mismatch_records)."""
    n_ok = 0
    n_mismatch = 0
    mismatches: List[Dict] = []

    for i, (cpp_c, py_c) in enumerate(zip(cpp_chunks, py_chunks)):
        chunk_errors: Dict[str, float] = {}
        for field, py_key in FIELD_MAP.items():
            cpp_val = cpp_c.get(field)
            py_val  = py_c.get(py_key)
            if cpp_val is None or py_val is None:
                continue
            err = abs(float(cpp_val) - float(py_val))
            tol = TOLERANCES.get(py_key, 1e-4)
            if err > tol:
                chunk_errors[field] = {  # type: ignore[assignment]
                    "cpp": cpp_val,
                    "py":  py_val,
                    "err": err,
                    "tol": tol,
                }

        dec_cpp = cpp_c.get("decision", "")
        dec_py  = py_c.get("decision", "")
        decision_match = (dec_cpp == dec_py)

        if chunk_errors or not decision_match:
            n_mismatch += 1
            mismatches.append({
                "chunk_idx":       i,
                "decision_cpp":    dec_cpp,
                "decision_py":     dec_py,
                "decision_match":  decision_match,
                "metric_errors":   chunk_errors,
            })
        else:
            n_ok += 1

    return n_ok, n_mismatch, mismatches


def run_parity(
    binary: str,
    labels_path: str,
    n_per_label: int,
    chunk_ms: int = 5000,
) -> Dict:
    """Run full parity check. Returns result dict."""
    files = select_files(labels_path, n_per_label)
    all_labels = sorted({e.get("label", "") for e in files})
    print(
        f"Selected {len(files)} audio files ({n_per_label}/label, "
        f"{len(all_labels)} labels: {', '.join(all_labels)})",
        file=sys.stderr,
    )

    chunk_sec = chunk_ms / 1000.0
    total_chunks  = 0
    total_ok      = 0
    total_mismatch = 0
    file_results: List[Dict] = []

    # Track worst-case error per metric across all chunks
    worst: Dict[str, float] = {k: 0.0 for k in FIELD_MAP}

    for entry in files:
        wav_path = gate_eval.resolve_path(entry["path"])
        name     = os.path.basename(wav_path)
        print(f"  {name:55s}", end="", file=sys.stderr, flush=True)

        # FLAC files cannot be read by the C++ binary (dr_wav is WAV-only).
        # Convert to a temporary WAV; Python evaluation uses the original path.
        tmp_wav: Optional[str] = None
        if not wav_path.lower().endswith(".wav"):
            tmp_wav = tempfile.mktemp(suffix=".wav")
            try:
                _flac_to_wav(wav_path, tmp_wav)
            except Exception as exc:
                print(f" [CONVERT FAILED: {exc}]", file=sys.stderr)
                file_results.append({"path": wav_path, "label": entry.get("label", ""), "status": "cpp_failed"})
                continue
        cpp_input = tmp_wav if tmp_wav else wav_path

        try:
            cpp_chunks = run_cpp_gate_only(binary, cpp_input, chunk_ms)
        finally:
            if tmp_wav:
                try:
                    os.unlink(tmp_wav)
                except OSError:
                    pass
        tmp_wav = None

        if cpp_chunks is None:
            print(" [C++ FAILED]", file=sys.stderr)
            file_results.append({"path": wav_path, "label": entry.get("label", ""), "status": "cpp_failed"})
            continue

        py_chunks = run_python_gate(wav_path, chunk_sec)

        # Chunk count mismatch is itself a failure
        n_cpp = len(cpp_chunks)
        n_py  = len(py_chunks)
        if n_cpp != n_py:
            print(f" [CHUNK COUNT MISMATCH: cpp={n_cpp}, py={n_py}]", file=sys.stderr)
            file_results.append({
                "path":   wav_path,
                "label":  entry.get("label", ""),
                "status": "chunk_count_mismatch",
                "n_cpp":  n_cpp,
                "n_py":   n_py,
            })
            total_mismatch += 1
            continue

        n_ok, n_mm, mms = compare_chunks(cpp_chunks, py_chunks)
        total_chunks   += n_cpp
        total_ok       += n_ok
        total_mismatch += n_mm

        # Update worst-case per-metric errors
        for mm in mms:
            for field, errd in mm["metric_errors"].items():
                if isinstance(errd, dict):
                    err = errd["err"]
                else:
                    err = errd
                if err > worst.get(field, 0.0):
                    worst[field] = err

        status = "ok" if n_mm == 0 else "mismatch"
        print(f" chunks={n_cpp} ok={n_ok} mismatch={n_mm}", file=sys.stderr)
        file_results.append({
            "path":      wav_path,
            "label":     entry.get("label", ""),
            "status":    status,
            "n_chunks":  n_cpp,
            "n_ok":      n_ok,
            "n_mismatch":n_mm,
            "mismatches":mms[:3],  # keep at most 3 per file to limit report size
        })

    # Overall pass/fail
    parity_ok = (total_mismatch == 0)

    # Per-label coverage summary
    labels_covered: Dict[str, Dict] = {}
    for fr in file_results:
        lbl = fr.get("label", "")
        if lbl not in labels_covered:
            labels_covered[lbl] = {"n_files": 0, "n_ok": 0, "n_mismatch": 0, "n_skip": 0}
        labels_covered[lbl]["n_files"] += 1
        if fr.get("status") == "ok":
            labels_covered[lbl]["n_ok"] += 1
        elif fr.get("status") in ("mismatch", "chunk_count_mismatch"):
            labels_covered[lbl]["n_mismatch"] += 1
        elif fr.get("status") == "cpp_failed":
            labels_covered[lbl]["n_skip"] += 1

    return {
        "n_files":          len(files),
        "n_chunks":         total_chunks,
        "n_ok":             total_ok,
        "n_mismatch":       total_mismatch,
        "parity_pass":      parity_ok,
        "worst_metric_err": worst,
        "tolerances":       TOLERANCES,
        "labels_covered":   labels_covered,
        "files":            file_results,
    }


def print_report(r: Dict) -> None:
    sep = "=" * 68
    print(f"\n{sep}")
    print("GATE PARITY REPORT  (Python vs C++)")
    print(sep)
    print(f"Files compared  : {r['n_files']}")
    print(f"Chunks compared : {r['n_chunks']}")
    print(f"Chunks OK       : {r['n_ok']}")
    print(f"Chunks mismatch : {r['n_mismatch']}")
    status = "PASS" if r["parity_pass"] else "FAIL"
    print(f"Parity status   : {status}")

    print()
    print("Worst-case absolute errors per metric (tolerance in parens):")
    for metric, worst in sorted(r["worst_metric_err"].items()):
        tol = r["tolerances"].get(metric, "?")
        flag = "  " if worst <= float(tol) else "!!"
        print(f"  {flag} {metric:20s}  worst={worst:.6f}  tol={tol}")

    lc = r.get("labels_covered", {})
    if lc:
        print()
        print("Labels covered:")
        all_expected = {
            "clean_speech", "clipped_or_distorted", "low_utility",
            "music", "speech_in_noise", "speech_in_reverb", "stationary_noise",
        }
        for lbl in sorted(all_expected):
            v = lc.get(lbl)
            if v is None:
                print(f"  {lbl:35s}  MISSING (no files selected)")
            else:
                n_skip = v.get("n_skip", 0)
                if v["n_mismatch"] > 0:
                    status = "MISMATCH"
                elif n_skip == v["n_files"]:
                    status = "SKIPPED (all cpp_failed)"
                elif n_skip > 0:
                    status = f"PARTIAL ({n_skip} skipped)"
                else:
                    status = "PASS"
                ok_pct = v["n_ok"] / max(v["n_files"] - n_skip, 1) if n_skip < v["n_files"] else 0.0
                print(
                    f"  {lbl:35s}  files={v['n_files']:2d}  "
                    f"ok={v['n_ok']:2d}  skip={n_skip:2d}  mismatch={v['n_mismatch']:2d}  "
                    f"{status}"
                    + (f"  ({ok_pct:.0%})" if status not in ("SKIPPED (all cpp_failed)",) else "")
                )

    mismatches = [
        f
        for f in r["files"]
        if f.get("status") in ("mismatch", "chunk_count_mismatch", "cpp_failed")
    ]
    if mismatches:
        print()
        print(f"Files with mismatches ({len(mismatches)}):")
        for f in mismatches[:10]:
            print(f"  {os.path.basename(f['path']):45s}  status={f['status']}")
            for mm in f.get("mismatches", []):
                if not mm["decision_match"]:
                    print(
                        f"    chunk {mm['chunk_idx']}: decision "
                        f"cpp={mm['decision_cpp']} py={mm['decision_py']}"
                    )
                for field, errd in mm.get("metric_errors", {}).items():
                    if isinstance(errd, dict):
                        print(
                            f"    chunk {mm['chunk_idx']}: {field}  "
                            f"cpp={errd['cpp']:.6f}  py={errd['py']:.6f}  "
                            f"err={errd['err']:.6f}  tol={errd['tol']}"
                        )
    print()


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--binary",
        default=DEFAULT_BINARY,
        help=f"C++ binary path (default: {DEFAULT_BINARY})",
    )
    ap.add_argument(
        "--labels",
        default=DEFAULT_LABELS,
        help=f"JSONL label file (default: {DEFAULT_LABELS})",
    )
    ap.add_argument(
        "--n",
        type=int,
        default=N_PER_LABEL,
        help=f"Files per label (default: {N_PER_LABEL})",
    )
    ap.add_argument(
        "--chunk-ms",
        type=int,
        default=5000,
        help="Chunk size in ms (must match C++ default, default: 5000)",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Directory to save parity JSON report (optional)",
    )
    args = ap.parse_args()

    # Resolve binary path relative to repo root when not absolute
    if not os.path.isabs(args.binary) and not os.path.exists(args.binary):
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
        candidate = os.path.join(repo_root, args.binary)
        if os.path.exists(candidate):
            args.binary = candidate

    if not os.path.exists(args.binary):
        print(f"ERROR: binary not found: {args.binary}", file=sys.stderr)
        print("Build with: bash scripts/build.sh Release cuda", file=sys.stderr)
        sys.exit(1)

    result = run_parity(args.binary, args.labels, args.n, args.chunk_ms)
    print_report(result)

    if args.out:
        import datetime
        os.makedirs(args.out, exist_ok=True)
        ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(args.out, f"parity_{ts}.json")
        with open(out_path, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"Parity report saved: {out_path}", file=sys.stderr)

    sys.exit(0 if result["parity_pass"] else 1)


if __name__ == "__main__":
    main()
