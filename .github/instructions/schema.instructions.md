---
applyTo: "runtime/cpp/src/logging/**,tools/python/eval/**,tools/python/tuning/**"
---

The TSV and JSON log schema is shared between C++ (logger.cpp) and Python readers.
Treat it as a versioned interface.

TSV fields (event=chunk):
  event, idx, start, end, dur, decision, reason,
  rms, silence, clip, zcr, flatness, centroid, rolloff, flux,
  bl, bm, bh, active, scene, infer_ms, [text], [asr_error]

JSON chunk fields:
  idx, start_sec, end_sec, decision, reason,
  rms, silence_ratio, clipping_ratio, zcr,
  flatness, centroid_hz, rolloff_hz, flux,
  band_low, band_mid, band_high, active_frac,
  scene, infer_ms, asr_ok, transcript

Stable reason strings (do not rename):
  rms_too_low, high_silence_ratio, high_clipping_ratio,
  stationary_noise_like, low_active_frame_fraction,
  weak_mid_band_speech_presence, excessive_high_band_energy,
  borderline_low_energy, borderline_noisy_speech, ok, gate_disabled

Rules:
- Additive JSON fields are safe. Removals and renames are breaking.
- If a field is renamed, update all Python eval scripts in the same patch.
- If a reason string changes, update gate_analysis.py REASON_DESCRIPTIONS in the same patch.
- Python readers must handle older logs that may be missing newer fields (use .get() with a default).
