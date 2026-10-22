/// @file
/// @brief Scene-history controller that retunes the rule gate and skips ASR.
/// @ingroup quality

#pragma once

#include <deque>

#include "edge_audio/quality/rule_gate.h"
#include "edge_audio/quality/scene_classifier.h"

namespace edge_audio {

/// @addtogroup quality
/// @{

/// @brief AdaptiveController parameters.
struct AdaptiveConfig {
    bool enabled = true;                 ///< When false the controller is a no-op.
    int history_chunks = 5;              ///< Length of the scene history window.
    double dominance_threshold = 0.60;   ///< Fraction of the window a label needs to dominate.
    double noise_rms_multiplier = 1.5;   ///< In a NOISE context, scale GateConfig::rms_min by this.
    double noise_flatness_tight = 0.70;  ///< In a NOISE context, cap flatness_max at this.
    bool music_skip_asr = true;          ///< Never transcribe chunks labelled MUSIC.
    bool silence_skip_asr = true;        ///< Never transcribe chunks labelled SILENCE.
};

/// @brief Uses recent scene labels to adjust how the next chunk is gated.
///
/// Call order per chunk: adapt_gate() → evaluate_chunk() → classify() →
/// skip_asr() → push_scene(). The adjusted config therefore always reflects
/// earlier chunks only.
class AdaptiveController {
public:
    /// @brief Start with an empty scene history.
    explicit AdaptiveController(const AdaptiveConfig& cfg);

    /// @brief Record the scene of the chunk just processed.
    void push_scene(SceneLabel label);

    /// @brief Most frequent label in the window if it reaches
    /// `dominance_threshold`, otherwise UNKNOWN.
    SceneLabel dominant_scene() const;

    /// @brief @p base, tightened when the window is dominated by NOISE.
    GateConfig adapt_gate(const GateConfig& base) const;

    /// @brief Whether ASR should be skipped for a chunk with label @p scene.
    bool skip_asr(SceneLabel scene) const;

    /// @brief Number of labels currently in the window.
    int window_size() const { return static_cast<int>(history_.size()); }

private:
    int count(SceneLabel label) const;

    AdaptiveConfig cfg_;
    std::deque<SceneLabel> history_;  // oldest first
};

/// @}

}  // namespace edge_audio
