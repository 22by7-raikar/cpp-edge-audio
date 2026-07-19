# Repository-wide Copilot instructions

## Project identity
Linux-first, C++-focused, deployment-oriented audio ML/STT pipeline.
ASR backend: whisper.cpp (git submodule at vendor/whisper.cpp).
Python is support tooling only (tools/python/).

## Teaching-first behavior
- Generate one small, coherent patch at a time unless the user explicitly asks for multi-file generation.
- Explain what each change does and why, briefly and technically.
- Prefer readable, traceable code over clever brevity.
- Do not silently make decisions that would affect benchmarks or deployed behavior.

## Architecture constraints
- Main runtime: runtime/cpp/ — C++17, CMake, no heavy external libraries beyond whisper.cpp.
- Python: tools/python/ — offline analysis, eval, tuning, packaging only.
- No iOS, Swift, Xcode, macOS-specific code.
- No diarization, source separation, emotion recognition, or GUI.
- No cloud-first or server-side inference paths.
- whisper.cpp remains the ASR backend unless explicitly told otherwise.

## Code generation rules
- Read existing files before editing them.
- One path change at a time — do not refactor unrelated code in the same patch.
- Do not invent APIs; verify signatures actually exist before calling them.
- Do not add speculative abstractions, helpers, or layers for one-off operations.
- Maintain C++ memory safety and correct alignment.
- When adding features to gate/chunker/scene/adaptive/logger — add a test in the same patch.

## Log schema compatibility
- The TSV and JSON log schema is consumed by tools/python/eval/*.py and tuning/sweep_thresholds.py.
- Do not add, rename, or remove log fields without updating all Python readers in the same patch.
- Additive JSON fields are safe. Removals and renames are breaking changes.
- Stable reason strings: rms_too_low, high_silence_ratio, high_clipping_ratio,
  stationary_noise_like, low_active_frame_fraction, weak_mid_band_speech_presence,
  excessive_high_band_energy, borderline_low_energy, borderline_noisy_speech, ok.

## Gate
- Interpretable, rule-based DSP gate. No ML in the gate yet.
- Returns PASS / FAIL / BORDERLINE with a reason string and GateMetrics struct.
- Features used: rms, silence_ratio, clipping_ratio, zcr, spectral_flatness,
  spectral_centroid, spectral_rolloff, spectral_flux, band_energy_low/mid/high, active_frame_frac.
- Keep defaults conservative. Prefer BORDERLINE over false FAILs on borderline speech.

## Benchmarking
- Treat benchmarking as a first-class feature.
- Always keep: model variant, thread count, chunk size, gate on/off, latency, RTF, accept rate.

## Style
- No emojis anywhere.
- No extra markdown docs unless explicitly requested.
- No decorative formatting.
- Short, technical comments.

## Strict non-goals
- No iOS, SwiftUI, Xcode.
- No ML gate yet.
- No microphone path until explicitly requested.
- No GUI, no cloud backend.
- no cloud-first implementation
- no diarization
- no speaker separation
- no emotion recognition as a main feature
- no fancy GUI
- no unnecessary docs

## Rate-limit aware workflow

- Do not perform broad repo audits unless explicitly asked.
- Do not run `find .`, `cat` large files, or inspect unrelated directories.
- Prefer targeted file reads.
- Work on one milestone per request.
- Avoid repeating project context already present in instructions.
- Keep summaries short and focused on changed files.
- Do not generate large reports in chat; write them to files and summarize key numbers.
- Before editing, inspect only the files listed in the task scope.