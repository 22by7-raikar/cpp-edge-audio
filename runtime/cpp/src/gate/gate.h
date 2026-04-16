#pragma once

#include <string>
#include "chunker/chunker.h"
#include "features.h"

namespace pipeline {

// -------------------------------------------------------
// Gate decision
// -------------------------------------------------------

enum class GateDecision { PASS, FAIL, BORDERLINE };

const char* gate_decision_str(GateDecision d);

// -------------------------------------------------------
// Feature metrics (populated by evaluate_chunk)
// -------------------------------------------------------
struct GateMetrics {
    double duration_sec   = 0.0;

    // Time-domain features
    double rms            = 0.0;
    double silence_ratio  = 0.0;   // fraction of samples below silence_thresh
    double clipping_ratio = 0.0;   // fraction of samples above clipping_thresh
    double zcr            = 0.0;   // zero-crossing rate (crossings / sec)

    // Spectral features (chunk-level aggregates from frame FFT pipeline)
    double spectral_flatness  = 0.0;  // mean flatness across frames [0,1]; near-1 = noise
    double spectral_centroid  = 0.0;  // mean centroid Hz
    double spectral_rolloff   = 0.0;  // mean rolloff Hz (85th percentile energy)
    double spectral_flux      = 0.0;  // mean frame-to-frame magnitude flux
    double band_energy_low    = 0.0;  // mean normalized energy 0-500 Hz
    double band_energy_mid    = 0.0;  // mean normalized energy 500-4000 Hz
    double band_energy_high   = 0.0;  // mean normalized energy 4000-Nyquist Hz
    double active_frame_frac  = 0.0;  // fraction of frames above rms activity threshold
};

// -------------------------------------------------------
// Gate result
// -------------------------------------------------------
struct GateResult {
    GateDecision decision = GateDecision::FAIL;
    std::string  reason;
    GateMetrics  metrics;
};

// -------------------------------------------------------
// Gate configuration (tunable thresholds)
// Defaults are conservative: prefer BORDERLINE over false FAILs.
// -------------------------------------------------------
struct GateConfig {
    // --- Time-domain thresholds ---
    double rms_min              = 0.003;   // below this -> BORDERLINE (borderline_low_energy)
    double rms_borderline_min   = 0.001;   // below this -> FAIL (rms_too_low)
    double max_silence_ratio    = 0.90;    // -> FAIL (high_silence_ratio)
    double max_clipping_ratio   = 0.05;    // -> FAIL (high_clipping_ratio)
    float  silence_thresh       = 0.005f;  // sample magnitude threshold for silence
    float  clipping_thresh      = 0.99f;   // sample magnitude threshold for clipping

    // ZCR: voiced speech ~60-200 cps/s; pure noise >> 300; silence ~0.
    // ZCR alone is a weak signal, only used to strengthen noise rejection.
    double zcr_max_noise        = 400.0;   // above this in combo with flatness -> FAIL (stationary_noise_like)

    // Active frame fraction: fraction of FFT frames above rms activity threshold.
    // Values near 0 mean almost all frames are silent/inactive.
    double min_active_frame_frac = 0.10;   // below this -> FAIL (low_active_frame_fraction)

    // --- Spectral flatness ---
    // Clean speech ~0.05-0.35; stationary noise ~0.7-1.0.
    double spectral_flatness_max  = 0.90;  // -> FAIL (stationary_noise_like)
    double spectral_flatness_warn = 0.72;  // -> BORDERLINE (borderline_noisy_speech)

    // --- Band energy thresholds (normalized fractions, sum ~1.0) ---
    // Speech typically: band_mid > 0.40, band_high < 0.55.
    // Weak mid-band suggests no voiced content.
    double min_band_mid           = 0.10;  // below this -> FAIL (weak_mid_band_speech_presence)
    // Very high-frequency dominance indicates noise, RF interference, or hiss.
    double max_band_high          = 0.70;  // above this -> FAIL (excessive_high_band_energy)

    // --- Frame config passed to the feature extractor ---
    FrameConfig frame;
};

// -------------------------------------------------------
// Evaluate a chunk through the gate.
// -------------------------------------------------------
GateResult evaluate_chunk(const Chunk& chunk, const GateConfig& cfg);

}  // namespace pipeline
