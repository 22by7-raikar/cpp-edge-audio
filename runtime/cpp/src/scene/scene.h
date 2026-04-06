#pragma once

#include <string>
#include "gate/gate.h"  // GateMetrics

namespace pipeline {

// -------------------------------------------------------
// Scene / context labels
// -------------------------------------------------------

enum class SceneLabel {
    SILENCE,            // very low energy, predominantly inactive
    NOISE,              // spectrally flat, broadband / background noise
    MUSIC,              // tonal, clearly non-speech energy distribution
    SPEECH,             // typical voice energy profile
    MIXED_SPEECH_NOISE, // speech with significant background noise
    UNKNOWN             // insufficient features (e.g. gate rejected before FFT)
};

const char* scene_label_str(SceneLabel s);

// -------------------------------------------------------
// Classification result
// -------------------------------------------------------
struct SceneResult {
    SceneLabel  label            = SceneLabel::UNKNOWN;
    double      confidence       = 0.0;  // 0-1 heuristic score for winning class
    std::string dominant_feature;        // which feature drove the decision
};

// -------------------------------------------------------
// Classification config (all thresholds tunable)
// -------------------------------------------------------
struct SceneConfig {
    // === SILENCE ===
    double silence_rms_max    = 0.003;  // RMS below this -> SILENCE
    double silence_active_max = 0.15;   // active_frame_frac below this -> SILENCE

    // === NOISE ===
    // Mean spectral flatness above this indicates broadband noise.
    // Clean speech ~0.05-0.35; white noise ~0.85-1.0; typical noise floor ~0.6-0.8.
    double noise_flatness_min = 0.65;

    // === MUSIC ===
    // Tonal, but band energy different from speech.
    double music_flatness_max  = 0.45;   // flatness must be low (tonal)
    double music_band_low_min  = 0.18;   // significant low-band energy
    double music_band_high_min = 0.08;   // significant high-band energy

    // === SPEECH ===
    // Dominant spectral energy in the mid-frequency band (500-4000 Hz)
    double speech_centroid_min  = 200.0;  // Hz — exclude sub-bass / hum only
    double speech_centroid_max  = 5000.0; // Hz — exclude very high pitched noise
    double speech_band_mid_dom  = 0.40;   // band_mid must exceed this fraction to be SPEECH

    // === MIXED_SPEECH_NOISE ===
    // If the flatness is elevated for something that would otherwise be SPEECH.
    double mixed_flatness_min = 0.40;
};

// -------------------------------------------------------
// Classify a chunk given its pre-computed gate metrics.
//
// The GateMetrics struct is used because it is always populated in main.cpp
// (even on FAIL decisions) and contains all time-domain + spectral features.
// Spectral fields will be zero if the gate short-circuited before FFT; the
// classifier handles this by falling through to SILENCE in that case.
// -------------------------------------------------------
SceneResult classify(const GateMetrics& m, const SceneConfig& cfg);

}  // namespace pipeline
