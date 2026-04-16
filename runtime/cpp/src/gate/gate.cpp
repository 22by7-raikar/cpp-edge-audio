// gate.cpp
// Gate evaluation: time-domain features (RMS, silence, clip, ZCR) plus
// FFT-based spectral features (flatness, centroid, rolloff, flux, band energies).
// Decision order: energy checks first (cheap), spectral checks second (FFT cost).

#include "gate.h"
#include "features.h"

#include <cmath>
#include <cstdlib>
#include <numeric>

namespace pipeline {

const char* gate_decision_str(GateDecision d) {
    switch (d) {
        case GateDecision::PASS:       return "PASS";
        case GateDecision::FAIL:       return "FAIL";
        case GateDecision::BORDERLINE: return "BORDERLINE";
    }
    return "UNKNOWN";
}

// -------------------------------------------------------
// Time-domain feature extraction
// -------------------------------------------------------

static GateMetrics compute_metrics(const Chunk& chunk, const GateConfig& cfg) {
    GateMetrics m;
    m.duration_sec = chunk.duration_sec();

    const auto& s = chunk.samples;
    if (s.empty()) return m;

    const size_t N = s.size();

    // RMS
    double sq_sum = 0.0;
    for (float v : s) sq_sum += static_cast<double>(v) * v;
    m.rms = std::sqrt(sq_sum / static_cast<double>(N));

    // Silence ratio and clipping ratio
    size_t silent_count  = 0;
    size_t clipped_count = 0;
    for (float v : s) {
        const float av = std::fabs(v);
        if (av < cfg.silence_thresh)  ++silent_count;
        if (av >= cfg.clipping_thresh) ++clipped_count;
    }
    m.silence_ratio  = static_cast<double>(silent_count)  / N;
    m.clipping_ratio = static_cast<double>(clipped_count) / N;

    // Zero-crossing rate (crossings per second)
    size_t zc = 0;
    for (size_t i = 1; i < N; ++i) {
        if ((s[i - 1] >= 0.0f) != (s[i] >= 0.0f)) ++zc;
    }
    m.zcr = (m.duration_sec > 0.0) ? static_cast<double>(zc) / m.duration_sec : 0.0;

    // ----- Spectral features via frame-level FFT pipeline -----
    const ChunkFeatures spec = extract_chunk_features(
        s.data(), static_cast<int>(N), chunk.sample_rate, cfg.frame);

    m.spectral_flatness  = spec.flatness_mean;
    m.spectral_centroid  = spec.centroid_mean;
    m.spectral_rolloff   = spec.rolloff_mean;
    m.spectral_flux      = spec.flux_mean;
    m.band_energy_low    = spec.band_low_mean;
    m.band_energy_mid    = spec.band_mid_mean;
    m.band_energy_high   = spec.band_high_mean;
    m.active_frame_frac  = spec.active_frame_frac;

    return m;
}

// -------------------------------------------------------
// Gate decision logic
// Ordered: cheap energy checks first, spectral checks last (FFT cost paid).
//
// Reason strings (stable — do not rename without updating schema.instructions.md
// and all Python eval scripts):
//   rms_too_low              : audio is essentially silent
//   high_silence_ratio       : too many silent samples
//   high_clipping_ratio      : severe clipping / ADC saturation
//   low_active_frame_fraction: almost no active FFT frames
//   stationary_noise_like    : high flatness [+high ZCR if extreme]
//   weak_mid_band_speech_presence: spectrum lacks mid-band voice energy
//   excessive_high_band_energy   : spectrum dominated by high-freq hiss/noise
//   borderline_low_energy    : RMS just above hard floor but below speech target
//   borderline_noisy_speech  : moderate flatness, could be speech in noise
//   ok                       : passed all checks
// -------------------------------------------------------

GateResult evaluate_chunk(const Chunk& chunk, const GateConfig& cfg) {
    GateResult result;
    result.metrics = compute_metrics(chunk, cfg);
    const GateMetrics& m = result.metrics;

    // ---- 1. Hard floor: almost entirely silent ----
    if (m.rms < cfg.rms_borderline_min) {
        result.decision = GateDecision::FAIL;
        result.reason   = "rms_too_low";
        return result;
    }

    // ---- 2. Too many silent samples ----
    if (m.silence_ratio > cfg.max_silence_ratio) {
        result.decision = GateDecision::FAIL;
        result.reason   = "high_silence_ratio";
        return result;
    }

    // ---- 3. Severe clipping ----
    if (m.clipping_ratio > cfg.max_clipping_ratio) {
        result.decision = GateDecision::FAIL;
        result.reason   = "high_clipping_ratio";
        return result;
    }

    // ---- 4. Almost no active FFT frames ----
    // active_frame_frac is 0 for purely time-domain-silent chunks, but can also
    // be near-zero for very low-level noise floors that pass the RMS floor above.
    if (m.active_frame_frac < cfg.min_active_frame_frac) {
        result.decision = GateDecision::FAIL;
        result.reason   = "low_active_frame_fraction";
        return result;
    }

    // ---- 5. Stationary noise: high spectral flatness ----
    // Flatness near 1.0 = white/pink noise; near 0 = tonal/speech.
    // Strengthen rejection further if ZCR is also abnormally high (noise-like).
    if (m.spectral_flatness > cfg.spectral_flatness_max) {
        result.decision = GateDecision::FAIL;
        result.reason   = "stationary_noise_like";
        return result;
    }
    if (m.spectral_flatness > cfg.spectral_flatness_warn &&
        m.zcr > cfg.zcr_max_noise) {
        // High flatness + high ZCR together strongly indicate stationary noise.
        result.decision = GateDecision::FAIL;
        result.reason   = "stationary_noise_like";
        return result;
    }

    // ---- 6. Weak mid-band speech presence ----
    // Speech concentrates energy in 500-4000 Hz. Very low band_mid suggests
    // content is not voice-like (HVAC hum in low-band only, or very high hiss).
    if (m.band_energy_mid < cfg.min_band_mid) {
        result.decision = GateDecision::FAIL;
        result.reason   = "weak_mid_band_speech_presence";
        return result;
    }

    // ---- 7. Excessive high-band energy ----
    // High-frequency dominance indicates noise, hiss, or RF interference.
    if (m.band_energy_high > cfg.max_band_high) {
        result.decision = GateDecision::FAIL;
        result.reason   = "excessive_high_band_energy";
        return result;
    }

    // ---- 8. BORDERLINE: low energy, not silent ----
    if (m.rms < cfg.rms_min) {
        result.decision = GateDecision::BORDERLINE;
        result.reason   = "borderline_low_energy";
        return result;
    }

    // ---- 9. BORDERLINE: moderate flatness (possible speech in noise) ----
    if (m.spectral_flatness > cfg.spectral_flatness_warn) {
        result.decision = GateDecision::BORDERLINE;
        result.reason   = "borderline_noisy_speech";
        return result;
    }

    // ---- 10. All checks passed ----
    result.decision = GateDecision::PASS;
    result.reason   = "ok";
    return result;
}

}  // namespace pipeline
