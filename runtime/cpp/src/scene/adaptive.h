#pragma once

#include <deque>
#include "scene/scene.h"
#include "gate/gate.h"

namespace pipeline {

// -------------------------------------------------------
// Adaptive controller configuration
// -------------------------------------------------------
struct AdaptiveConfig {
    bool enabled       = true;

    // Rolling window length (in chunks) for scene smoothing.
    // The controller considers the dominant scene over this window
    // before modifying inference behavior.
    int history_chunks = 5;

    // === NOISE context adjustments ===
    // When the rolling window is predominantly NOISE, tighten gate thresholds
    // to avoid sending low-quality chunks to ASR.
    double noise_rms_multiplier    = 1.5;   // multiply base rms_min
    double noise_flatness_tight    = 0.70;  // cap flatness_max at this value

    // === MUSIC context ===
    // Music is not transcribable; skip ASR entirely for these chunks.
    bool music_skip_asr   = true;

    // === SILENCE context ===
    // Already caught by gate, but double-gate in adaptive layer.
    bool silence_skip_asr = true;

    // Minimum fraction of window that must match a label for it
    // to be considered "dominant" and trigger context adaptation.
    double dominance_threshold = 0.60;  // 60% of window
};

// -------------------------------------------------------
// Adaptive controller
//
// Maintains a rolling window of recent scene labels and provides:
//   - dominant_scene(): the most common scene in the window
//   - adapt_gate():     returns a modified GateConfig for next chunk
//   - skip_asr():       whether ASR should be suppressed for a scene
//
// Designed for the streaming case: push_scene() is called AFTER evaluating
// a chunk, so the returned adapted config applies to the NEXT chunk.
// -------------------------------------------------------
class AdaptiveController {
public:
    explicit AdaptiveController(const AdaptiveConfig& cfg);

    // Push the scene classification for the most recently processed chunk.
    void push_scene(SceneLabel label);

    // Return the most frequent scene in the current window.
    // Returns SceneLabel::UNKNOWN if the window is empty or no scene
    // exceeds the dominance_threshold.
    SceneLabel dominant_scene() const;

    // Return a GateConfig derived from base by applying context-aware
    // threshold adjustments.  If adaptive is disabled, returns base unchanged.
    GateConfig adapt_gate(const GateConfig& base) const;

    // Returns true if ASR should be skipped for the given scene.
    // This is checked AFTER gate evaluation, acting as a higher-level
    // semantic filter on top of the acoustic gate.
    bool skip_asr(SceneLabel scene) const;

    // Current window size (number of chunks observed so far).
    int window_size() const { return static_cast<int>(history_.size()); }

private:
    AdaptiveConfig           cfg_;
    std::deque<SceneLabel>   history_;  // oldest at front

    // Count occurrences of each label in current window.
    int count(SceneLabel label) const;
};

}  // namespace pipeline
