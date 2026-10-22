/// @file
/// @brief Rule-based acoustic scene labels from gate metrics.
/// @ingroup quality

#pragma once

#include <string>

#include "edge_audio/quality/rule_gate.h"

namespace edge_audio {

/// @addtogroup quality
/// @{

/// @brief Coarse acoustic scene.
enum class SceneLabel {
    SILENCE,             ///< Very low energy or few active frames.
    NOISE,               ///< Spectrally flat, broadband noise.
    MUSIC,               ///< Tonal with strong low- and high-band energy.
    SPEECH,              ///< Voice-band dominant, centroid in the voice range.
    MIXED_SPEECH_NOISE,  ///< Speech-like with elevated flatness.
    UNKNOWN,             ///< No rule matched, or spectral metrics are missing.
};

/// @brief Stable upper-case name (`MIXED` for MIXED_SPEECH_NOISE).
const char* scene_label_str(SceneLabel s);

/// @brief A scene label with a heuristic confidence.
struct SceneResult {
    SceneLabel label = SceneLabel::UNKNOWN;  ///< Winning label.
    double confidence = 0.0;                 ///< Heuristic score in [0, 1]; not a probability.
    std::string dominant_feature;            ///< Feature that decided the label.
};

/// @brief Scene rule thresholds.
struct SceneConfig {
    double silence_rms_max = 0.003;       ///< SILENCE below this RMS...
    double silence_active_max = 0.15;     ///< ...or below this active-frame fraction.
    double noise_flatness_min = 0.65;     ///< NOISE at or above this flatness.
    double music_flatness_max = 0.45;     ///< MUSIC requires flatness below this,
    double music_band_low_min = 0.18;     ///< low-band energy at least this,
    double music_band_high_min = 0.08;    ///< and high-band energy at least this.
    double speech_centroid_min = 200.0;   ///< SPEECH centroid lower bound, Hz.
    double speech_centroid_max = 5000.0;  ///< SPEECH centroid upper bound, Hz.
    double speech_band_mid_dom = 0.40;    ///< SPEECH requires voice-band fraction at least this.
    double mixed_flatness_min = 0.40;     ///< Speech-like with flatness above this is MIXED.
};

/// @brief Label a chunk from its gate metrics. First matching rule wins:
/// SILENCE, NOISE, MUSIC, SPEECH, MIXED_SPEECH_NOISE, otherwise UNKNOWN.
SceneResult classify(const GateMetrics& m, const SceneConfig& cfg);

/// @}

}  // namespace edge_audio
