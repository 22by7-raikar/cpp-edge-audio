// scene.cpp
// Rule-based scene classifier using pre-computed gate metrics.
//
// Decision priority (first rule that matches wins):
//   1. SILENCE       - very low RMS or inactive frames
//   2. NOISE         - spectrally flat (broadband)
//   3. MUSIC         - tonal with significant low+high band energy
//   4. SPEECH        - mid-band dominant, centroid in voice range
//   5. MIXED         - speech-like but elevated flatness (noisy conditions)
//   6. UNKNOWN       - fallback if spectral features are unavailable

#include "scene/scene.h"

#include <cmath>
#include <algorithm>

namespace pipeline {

const char* scene_label_str(SceneLabel s) {
    switch (s) {
        case SceneLabel::SILENCE:            return "SILENCE";
        case SceneLabel::NOISE:              return "NOISE";
        case SceneLabel::MUSIC:              return "MUSIC";
        case SceneLabel::SPEECH:             return "SPEECH";
        case SceneLabel::MIXED_SPEECH_NOISE: return "MIXED";
        case SceneLabel::UNKNOWN:            return "UNKNOWN";
    }
    return "UNKNOWN";
}

// -------------------------------------------------------
// Internal helpers
// -------------------------------------------------------

// Soft confidence: how far a feature value is from its threshold, clamped [0,1].
// dir=+1 means higher value = higher confidence (e.g. flatness above noise threshold)
// dir=-1 means lower value = higher confidence (e.g. rms below silence threshold)
static double soft_conf(double val, double threshold, double spread, int dir) {
    double raw = dir * (val - threshold) / std::max(spread, 1e-9);
    return std::min(1.0, std::max(0.0, 0.5 + 0.5 * std::tanh(raw * 3.0)));
}

// -------------------------------------------------------
// classify
// -------------------------------------------------------
SceneResult classify(const GateMetrics& m, const SceneConfig& cfg) {
    SceneResult result;

    const double flatness   = m.spectral_flatness;
    const double centroid   = m.spectral_centroid;
    const double band_low   = m.band_energy_low;
    const double band_mid   = m.band_energy_mid;
    const double band_high  = m.band_energy_high;
    const double active     = m.active_frame_frac;
    const double rms        = m.rms;

    // -----------------------------------------------------------
    // 1. SILENCE
    // Very low RMS or very few active frames.
    // These chunks are not speech regardless of spectral content.
    // -----------------------------------------------------------
    if (rms < cfg.silence_rms_max || active < cfg.silence_active_max) {
        result.label            = SceneLabel::SILENCE;
        result.dominant_feature = (rms < cfg.silence_rms_max) ? "rms" : "active_frac";
        result.confidence       = soft_conf(rms, cfg.silence_rms_max, cfg.silence_rms_max, -1);
        return result;
    }

    // If spectral features were not computed (gate short-circuited before FFT),
    // flatness will be 0. Flag as UNKNOWN so the pipeline doesn't misclassify
    // as speech just because centroid is 0.
    if (flatness == 0.0 && centroid == 0.0 && band_mid == 0.0) {
        result.label            = SceneLabel::UNKNOWN;
        result.dominant_feature = "no_spectral_features";
        result.confidence       = 0.0;
        return result;
    }

    // -----------------------------------------------------------
    // 2. NOISE
    // High spectral flatness indicates broadband / stationary noise.
    // -----------------------------------------------------------
    if (flatness >= cfg.noise_flatness_min) {
        result.label            = SceneLabel::NOISE;
        result.dominant_feature = "flatness";
        result.confidence       = soft_conf(flatness,
                                            cfg.noise_flatness_min,
                                            1.0 - cfg.noise_flatness_min,
                                            +1);
        return result;
    }

    // -----------------------------------------------------------
    // 3. MUSIC
    // Tonal (low flatness) but energy spreads across low + high bands,
    // which is distinct from narrowband voiced speech.
    // -----------------------------------------------------------
    const bool is_tonal        = flatness < cfg.music_flatness_max;
    const bool has_low_energy  = band_low  >= cfg.music_band_low_min;
    const bool has_high_energy = band_high >= cfg.music_band_high_min;

    if (is_tonal && has_low_energy && has_high_energy) {
        result.label            = SceneLabel::MUSIC;
        result.dominant_feature = "band_energy_low+high";
        // Confidence: how well it satisfies all three criteria
        double s_tonal = soft_conf(flatness, cfg.music_flatness_max,
                                   cfg.music_flatness_max, -1);
        double s_low   = soft_conf(band_low,  cfg.music_band_low_min,
                                   cfg.music_band_low_min, +1);
        double s_high  = soft_conf(band_high, cfg.music_band_high_min,
                                   cfg.music_band_high_min, +1);
        result.confidence = std::cbrt(s_tonal * s_low * s_high);  // geometric mean
        return result;
    }

    // -----------------------------------------------------------
    // 4. SPEECH
    // Mid-band dominant, centroid in voice range, low flatness.
    // -----------------------------------------------------------
    const bool centroid_in_range = centroid >= cfg.speech_centroid_min &&
                                   centroid <= cfg.speech_centroid_max;
    const bool mid_dominant      = band_mid >= cfg.speech_band_mid_dom;
    const bool flatness_ok       = flatness < cfg.mixed_flatness_min;

    if (centroid_in_range && mid_dominant && flatness_ok) {
        result.label            = SceneLabel::SPEECH;
        result.dominant_feature = "band_mid+centroid";
        // Higher confidence when mid band is clearly dominant
        result.confidence       = soft_conf(band_mid,
                                            cfg.speech_band_mid_dom,
                                            1.0 - cfg.speech_band_mid_dom,
                                            +1);
        return result;
    }

    // -----------------------------------------------------------
    // 5. MIXED_SPEECH_NOISE
    // Broadly speech-like but flatness is elevated, suggesting
    // background noise is present alongside a voice signal.
    // -----------------------------------------------------------
    if (centroid_in_range && mid_dominant && flatness >= cfg.mixed_flatness_min) {
        result.label            = SceneLabel::MIXED_SPEECH_NOISE;
        result.dominant_feature = "flatness_elevated";
        result.confidence       = soft_conf(flatness,
                                            cfg.mixed_flatness_min,
                                            cfg.noise_flatness_min - cfg.mixed_flatness_min,
                                            +1);
        return result;
    }

    // -----------------------------------------------------------
    // 6. UNKNOWN — features present but don't match any pattern cleanly
    // -----------------------------------------------------------
    result.label            = SceneLabel::UNKNOWN;
    result.dominant_feature = "no_clear_pattern";
    result.confidence       = 0.0;
    return result;
}

}  // namespace pipeline
