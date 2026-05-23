"""bench_vad_asr.py — Compare fixed-window vs VAD-based ASR.

Drives the audio_pipeline C++ binary in three modes per file:
  fixed:    fixed-window chunking + gate  (default pipeline)
  vad_gate: VAD segmentation + gate
  vad_open: VAD segmentation, no gate

Reports per-mode aggregate metrics:
  files, audio_sec, asr_sec, removed_pct, n_segments, n_asr_calls,
  wall_ms, infer_ms, rtf, accept_rate, backend, model

For clean_speech entries, optionally computes WER if LibriSpeech
transcripts are available on disk.

Usage:
    python tools/python/eval/bench_vad_asr.py \\
        --model vendor/whisper.cpp/models/ggml-base.en.bin \\
        [--labels data/labels/eval_subset.jsonl] \\
        [--binary runtime/cpp/build/audio_pipeline] \\
        [--max-files 5] \\
        [--threads 4] \\
        [--chunk-ms 5000] \\
        [--out-dir benchmarks/results/vad_asr]

Requires:
    audio_pipeline binary pre-built with --vad-asr support.
    Build:  cd runtime/cpp && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

_REPO_ROOT = Path(__file__).resolve().parents[3]

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def resolve_path(p: str) -> str:
    pp = Path(p)
    return str(pp if pp.is_absolute() else _REPO_ROOT / pp)


def load_labels(path: str) -> List[Dict]:
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# LibriSpeech transcript index
# ---------------------------------------------------------------------------

def load_librispeech_transcripts() -> Dict[str, str]:
    """Walk all librispeech splits under data/raw/librispeech and index utterance→text."""
    trans: Dict[str, str] = {}
    ls_root = _REPO_ROOT / "data" / "raw" / "librispeech"
    if not ls_root.exists():
        return trans
    for txt in ls_root.rglob("*.trans.txt"):
        try:
            with open(txt) as f:
                for line in f:
                    line = line.strip()
                    if not line or " " not in line:
                        continue
                    sep = line.index(" ")
                    trans[line[:sep]] = line[sep + 1:]
        except OSError:
            pass
    return trans


# ---------------------------------------------------------------------------
# WER
# ---------------------------------------------------------------------------

def _edit_distance(a: List[str], b: List[str]) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if a[i - 1] == b[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def word_error_rate(ref: str, hyp: str) -> Optional[float]:
    ref_w = ref.upper().split()
    hyp_w = hyp.upper().split()
    if not ref_w:
        return None
    return _edit_distance(ref_w, hyp_w) / len(ref_w)


# ---------------------------------------------------------------------------
# Audio → temp WAV  (binary only reads WAV via dr_wav)
# ---------------------------------------------------------------------------

def audio_to_wav(src_path: str, tmpdir: str) -> str:
    """Load audio with soundfile, write float32 mono 16kHz WAV, return path."""
    data, sr = sf.read(src_path, dtype="float32", always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1)
    # Resample to 16kHz if needed (binary resamples too, but let's keep it clean)
    # soundfile can't resample; just pass the native rate and let the binary handle it.
    out_path = os.path.join(tmpdir, os.path.basename(src_path) + ".wav")
    sf.write(out_path, data, sr, subtype="FLOAT")
    return out_path


# ---------------------------------------------------------------------------
# Binary runner
# ---------------------------------------------------------------------------

MODES: List[Tuple[str, List[str], str]] = [
    ("fixed",    [],                         "fixed-window + gate"),
    ("vad_gate", ["--vad-asr"],              "VAD + gate"),
    ("vad_open", ["--vad-asr", "--no-gate"], "VAD, no gate"),
]


def run_mode(binary: str, wav_path: str, model: str, threads: int,
             chunk_ms: int, extra_flags: List[str]) -> Dict:
    """Run audio_pipeline in one mode; return parsed result dict."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as jf:
        json_path = jf.name

    cmd = [
        binary,
        "--input",      wav_path,
        "--model",      model,
        "--threads",    str(threads),
        "--chunk-ms",   str(chunk_ms),
        "--bench-json", json_path,
    ] + extra_flags

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall_ms = (time.perf_counter() - t0) * 1000.0

    result: Dict = {"wall_ms": wall_ms, "ok": False, "backend": "unknown"}

    # Parse backend from stderr
    for line in proc.stderr.splitlines():
        if "backend_active=" in line:
            for tok in line.split():
                if tok.startswith("backend_active="):
                    result["backend"] = tok.split("=", 1)[1]
                    break

    try:
        with open(json_path) as jf:
            data = json.load(jf)

        summary = data.get("summary", {})
        chunks  = data.get("chunks", [])

        asr_sec = sum(
            c["end_sec"] - c["start_sec"]
            for c in chunks
            if c.get("decision") in ("PASS", "BORDERLINE")
        )
        transcript = " ".join(
            c.get("transcript", "").strip()
            for c in chunks
            if c.get("transcript", "").strip()
        )

        result.update({
            "ok":           True,
            "audio_sec":    summary.get("audio_sec",        0.0),
            "n_segments":   summary.get("total_chunks",     0),
            "n_asr_calls":  summary.get("passed", 0) + summary.get("borderline", 0),
            "infer_ms":     summary.get("total_infer_ms",   0.0),
            "accept_rate":  summary.get("accept_rate",      0.0),
            "asr_sec":      asr_sec,
            "transcript":   transcript,
            "stderr":       proc.stderr,
        })
        # RTF normalised to total audio duration
        audio_sec = result["audio_sec"]
        result["rtf"] = (result["infer_ms"] / (audio_sec * 1000.0)
                         if audio_sec > 0 else 0.0)
    except Exception as exc:
        result["error"] = str(exc)
        result["stderr_tail"] = proc.stderr[-400:] if proc.stderr else ""
    finally:
        try:
            os.unlink(json_path)
        except OSError:
            pass

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model",     required=True,
                    help="Path to whisper.cpp model .bin file")
    ap.add_argument("--labels",    default="data/labels/eval_subset.jsonl",
                    help="JSONL label file (default: data/labels/eval_subset.jsonl)")
    ap.add_argument("--binary",    default="runtime/cpp/build/audio_pipeline",
                    help="audio_pipeline binary (default: runtime/cpp/build/audio_pipeline)")
    ap.add_argument("--max-files", type=int, default=5,
                    help="Max files per label (default: 5)")
    ap.add_argument("--threads",   type=int, default=4)
    ap.add_argument("--chunk-ms",  type=int, default=5000)
    ap.add_argument("--out-dir",   default="benchmarks/results/vad_asr",
                    help="Output directory (default: benchmarks/results/vad_asr)")
    ap.add_argument("--labels-filter", nargs="*",
                    help="Only process these labels (e.g. clean_speech low_utility)")
    args = ap.parse_args()

    binary = resolve_path(args.binary)
    model  = resolve_path(args.model)
    labels_path = resolve_path(args.labels)
    out_dir = Path(resolve_path(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    if not os.path.isfile(binary):
        print(f"ERROR: binary not found: {binary}", file=sys.stderr)
        print("Build with:", file=sys.stderr)
        print("  cd runtime/cpp && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release "
              "-DGGML_CUDA=ON && cmake --build build -j", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(model):
        print(f"ERROR: model not found: {model}", file=sys.stderr)
        sys.exit(1)

    # Load labels
    entries = load_labels(labels_path)
    by_label: Dict[str, List[Dict]] = defaultdict(list)
    for e in entries:
        by_label[e.get("label", "unknown")].append(e)

    if args.labels_filter:
        by_label = {k: v for k, v in by_label.items() if k in args.labels_filter}

    # Cap per label
    by_label = {k: v[: args.max_files] for k, v in by_label.items()}
    total_files = sum(len(v) for v in by_label.values())

    print(f"Binary       : {binary}")
    print(f"Model        : {model}")
    print(f"Labels file  : {labels_path}")
    print(f"Total files  : {total_files}  ({args.max_files} per label)")
    print(f"Chunk ms     : {args.chunk_ms}")
    print(f"Threads      : {args.threads}")
    print(f"Output dir   : {out_dir}")
    print()

    # Load LibriSpeech transcripts (best-effort)
    print("Indexing LibriSpeech transcripts...", end=" ", flush=True)
    transcripts = load_librispeech_transcripts()
    print(f"{len(transcripts)} utterances found")
    print()

    # Per-mode accumulators: mode_key → {field → total}
    mode_acc: Dict[str, Dict] = {mk: defaultdict(float) for mk, _, _ in MODES}
    # Per-file WER list: mode_key → [wer_value]
    mode_wer: Dict[str, List[float]] = {mk: [] for mk, _, _ in MODES}
    # Per-file results for TSV
    per_file_rows: List[Dict] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        file_num = 0
        for label in sorted(by_label):
            for entry in by_label[label]:
                file_num += 1
                src_path = resolve_path(entry["path"])
                utt_id   = entry.get("base_utterance_id", "")
                ref_text = transcripts.get(utt_id, "") if utt_id else ""

                # Write temp WAV (binary uses dr_wav, no FLAC support)
                try:
                    wav_path = audio_to_wav(src_path, tmpdir)
                except Exception as exc:
                    print(f"  [{file_num}/{total_files}] SKIP {os.path.basename(src_path)}: {exc}",
                          file=sys.stderr)
                    continue

                print(f"  [{file_num}/{total_files}] {label} / {os.path.basename(src_path)}")

                file_row: Dict = {"path": entry["path"], "label": label, "utt_id": utt_id}
                for mode_key, extra_flags, mode_desc in MODES:
                    r = run_mode(binary, wav_path, model, args.threads,
                                 args.chunk_ms, extra_flags)

                    if not r["ok"]:
                        err = r.get("error", "unknown")
                        print(f"    {mode_key}: FAILED — {err}", file=sys.stderr)
                        if "stderr_tail" in r:
                            print(f"    stderr: {r['stderr_tail']}", file=sys.stderr)
                        continue

                    # Accumulate
                    acc = mode_acc[mode_key]
                    acc["n_files"]    += 1
                    acc["audio_sec"]  += r["audio_sec"]
                    acc["asr_sec"]    += r["asr_sec"]
                    acc["n_segments"] += r["n_segments"]
                    acc["n_asr_calls"]+= r["n_asr_calls"]
                    acc["infer_ms"]   += r["infer_ms"]
                    acc["wall_ms"]    += r["wall_ms"]

                    # WER
                    wer_val: Optional[float] = None
                    if ref_text and r.get("transcript"):
                        wer_val = word_error_rate(ref_text, r["transcript"])
                        mode_wer[mode_key].append(wer_val)

                    file_row[f"{mode_key}_ok"]         = r["ok"]
                    file_row[f"{mode_key}_audio_sec"]  = r["audio_sec"]
                    file_row[f"{mode_key}_asr_sec"]    = r["asr_sec"]
                    file_row[f"{mode_key}_n_segs"]     = r["n_segments"]
                    file_row[f"{mode_key}_n_asr"]      = r["n_asr_calls"]
                    file_row[f"{mode_key}_infer_ms"]   = r["infer_ms"]
                    file_row[f"{mode_key}_wall_ms"]    = r["wall_ms"]
                    file_row[f"{mode_key}_wer"]        = wer_val
                    file_row[f"{mode_key}_backend"]    = r["backend"]
                    file_row[f"{mode_key}_transcript"] = r.get("transcript", "")

                    removed_pct = (1.0 - r["asr_sec"] / r["audio_sec"]) * 100.0 \
                        if r["audio_sec"] > 0 else 0.0
                    wer_str = f"  WER={wer_val:.3f}" if wer_val is not None else ""
                    print(f"    {mode_key:<9}  segs={r['n_segments']:>3}  "
                          f"asr_calls={r['n_asr_calls']:>3}  "
                          f"audio={r['audio_sec']:.1f}s  asr_sent={r['asr_sec']:.1f}s  "
                          f"removed={removed_pct:.0f}%  "
                          f"infer={r['infer_ms']:.0f}ms  wall={r['wall_ms']:.0f}ms  "
                          f"backend={r['backend']}{wer_str}")

                per_file_rows.append(file_row)

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print()
    print("=" * 90)
    print("AGGREGATE RESULTS")
    print("=" * 90)
    hdr = (f"{'mode':<12}  {'files':>5}  {'audio_s':>7}  {'asr_s':>7}  "
           f"{'removed%':>8}  {'n_segs':>6}  {'asr_calls':>9}  "
           f"{'infer_ms':>8}  {'rtf':>6}  {'wall_ms':>8}  {'wer':>6}  backend")
    print(hdr)
    print("-" * len(hdr))

    # backend from the last seen file
    mode_backend = {mk: "?" for mk, _, _ in MODES}
    for row in per_file_rows:
        for mk, _, _ in MODES:
            b = row.get(f"{mk}_backend", "?")
            if b and b != "unknown":
                mode_backend[mk] = b

    summary_rows = []
    for mode_key, _, mode_desc in MODES:
        acc = mode_acc[mode_key]
        if acc["n_files"] == 0:
            continue
        audio = acc["audio_sec"]
        asr   = acc["asr_sec"]
        removed_pct = (1.0 - asr / audio) * 100.0 if audio > 0 else 0.0
        infer_ms = acc["infer_ms"]
        rtf = infer_ms / (audio * 1000.0) if audio > 0 else 0.0
        wall_ms = acc["wall_ms"]
        wers = mode_wer[mode_key]
        avg_wer = sum(wers) / len(wers) if wers else None
        wer_str = f"{avg_wer:.3f}" if avg_wer is not None else "N/A"
        backend = mode_backend[mode_key]

        print(f"{mode_key:<12}  {int(acc['n_files']):>5}  {audio:>7.1f}  {asr:>7.1f}  "
              f"{removed_pct:>7.1f}%  {int(acc['n_segments']):>6}  {int(acc['n_asr_calls']):>9}  "
              f"{infer_ms:>8.0f}  {rtf:>6.4f}  {wall_ms:>8.0f}  {wer_str:>6}  {backend}")

        summary_rows.append({
            "mode":        mode_key,
            "mode_desc":   mode_desc,
            "n_files":     int(acc["n_files"]),
            "audio_sec":   round(audio, 2),
            "asr_sec":     round(asr, 2),
            "removed_pct": round(removed_pct, 2),
            "n_segments":  int(acc["n_segments"]),
            "n_asr_calls": int(acc["n_asr_calls"]),
            "infer_ms":    round(infer_ms, 1),
            "rtf":         round(rtf, 5),
            "wall_ms":     round(wall_ms, 1),
            "avg_wer":     round(avg_wer, 4) if avg_wer is not None else None,
            "backend":     backend,
            "model":       os.path.basename(model),
        })
    print("-" * len(hdr))

    # -----------------------------------------------------------------------
    # Key comparison: VAD savings vs fixed
    # -----------------------------------------------------------------------
    fixed_row = next((r for r in summary_rows if r["mode"] == "fixed"), None)
    vadg_row  = next((r for r in summary_rows if r["mode"] == "vad_gate"), None)
    if fixed_row and vadg_row:
        audio_saved_sec = fixed_row["asr_sec"] - vadg_row["asr_sec"]
        audio_saved_pct = audio_saved_sec / fixed_row["asr_sec"] * 100.0 \
            if fixed_row["asr_sec"] > 0 else 0.0
        infer_delta_ms  = vadg_row["infer_ms"] - fixed_row["infer_ms"]
        infer_delta_pct = infer_delta_ms / fixed_row["infer_ms"] * 100.0 \
            if fixed_row["infer_ms"] > 0 else 0.0
        print()
        print(f"VAD+gate vs fixed:")
        print(f"  Audio to ASR:     {fixed_row['asr_sec']:.1f}s (fixed)  vs  "
              f"{vadg_row['asr_sec']:.1f}s (VAD+gate)  "
              f"({audio_saved_sec:.1f}s less, {audio_saved_pct:.1f}% reduction)")
        if fixed_row["infer_ms"] > 0:
            direction = "saved" if infer_delta_ms < 0 else "overhead"
            print(f"  Inference time:   {fixed_row['infer_ms']:.0f}ms (fixed)  vs  "
                  f"{vadg_row['infer_ms']:.0f}ms (VAD+gate)  "
                  f"({abs(infer_delta_ms):.0f}ms {direction}, "
                  f"{abs(infer_delta_pct):.1f}% {'faster' if infer_delta_ms < 0 else 'slower'})")
            print(f"  RTF:              {fixed_row['rtf']:.4f} (fixed)  vs  "
                  f"{vadg_row['rtf']:.4f} (VAD+gate)")
        if fixed_row.get("avg_wer") is not None and vadg_row.get("avg_wer") is not None:
            wer_delta = vadg_row["avg_wer"] - fixed_row["avg_wer"]
            print(f"  WER (clean):      {fixed_row['avg_wer']:.3f} (fixed)  vs  "
                  f"{vadg_row['avg_wer']:.3f} (VAD+gate)  "
                  f"(delta={wer_delta:+.3f})")

    # -----------------------------------------------------------------------
    # Save outputs
    # -----------------------------------------------------------------------
    import datetime
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = Path(model).stem.replace("ggml-", "")

    # Per-file TSV
    per_file_tsv = out_dir / f"per_file_{model_tag}_{ts}.tsv"
    if per_file_rows:
        # Collect all column names
        cols = list(per_file_rows[0].keys())
        with open(per_file_tsv, "w") as fh:
            fh.write("\t".join(cols) + "\n")
            for row in per_file_rows:
                fh.write("\t".join(str(row.get(c, "")) for c in cols) + "\n")

    # Summary JSON
    summary_json = out_dir / f"summary_{model_tag}_{ts}.json"
    with open(summary_json, "w") as fh:
        json.dump({
            "timestamp": ts,
            "model":     os.path.basename(model),
            "labels":    os.path.basename(labels_path),
            "max_files_per_label": args.max_files,
            "chunk_ms":  args.chunk_ms,
            "threads":   args.threads,
            "modes":     summary_rows,
        }, fh, indent=2)

    print()
    print(f"Per-file TSV  : {per_file_tsv}")
    print(f"Summary JSON  : {summary_json}")


if __name__ == "__main__":
    main()
