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
// -------------------------------------------------------

GateResult evaluate_chunk(const Chunk& chunk, const GateConfig& cfg) {
    GateResult result;
    result.metrics = compute_metrics(chunk, cfg);
    const GateMetrics& m = result.metrics;

    // FAIL: almost entirely silent
    if (m.rms < cfg.rms_borderline_min) {
        result.decision = GateDecision::FAIL;
        result.reason   = "rms_too_low";
        return result;
    }

    // FAIL: too many silent frames
    if (m.silence_ratio > cfg.max_silence_ratio) {
        result.decision = GateDecision::FAIL;
        result.reason   = "high_silence_ratio";
        return result;
    }

    // FAIL: severe clipping
    if (m.clipping_ratio > cfg.max_clipping_ratio) {
        result.decision = GateDecision::FAIL;
        result.reason   = "high_clipping_ratio";
        return result;
    }

    // BORDERLINE: low energy but not silent
    if (m.rms < cfg.rms_min) {
        result.decision = GateDecision::BORDERLINE;
        result.reason   = "low_rms";
        return result;
    }

    // FAIL: flatness dominated by stationary noise (white/pink noise, HVAC, etc.)
    if (m.spectral_flatness > cfg.spectral_flatness_max) {
        result.decision = GateDecision::FAIL;
        result.reason   = "high_spectral_flatness";
        return result;
    }

    // BORDERLINE: moderately elevated flatness (may be speech in noise)
    if (m.spectral_flatness > cfg.spectral_flatness_warn) {
        result.decision = GateDecision::BORDERLINE;
        result.reason   = "elevated_flatness";
        return result;
    }

    result.decision = GateDecision::PASS;
    result.reason   = "ok";
    return result;
}

}  // namespace pipeline
