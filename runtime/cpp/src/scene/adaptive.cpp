// adaptive.cpp
// Adaptive controller: adjusts gate thresholds and ASR routing based on
// a rolling window of scene classifications. Designed for the streaming
// inference path where scene context from recent chunks informs decisions
// on the next chunk.

#include "scene/adaptive.h"

#include <algorithm>

namespace pipeline {

AdaptiveController::AdaptiveController(const AdaptiveConfig& cfg)
    : cfg_(cfg) {}

void AdaptiveController::push_scene(SceneLabel label) {
    history_.push_back(label);
    while (static_cast<int>(history_.size()) > cfg_.history_chunks) {
        history_.pop_front();
    }
}

int AdaptiveController::count(SceneLabel label) const {
    int n = 0;
    for (SceneLabel s : history_) {
        if (s == label) ++n;
    }
    return n;
}

SceneLabel AdaptiveController::dominant_scene() const {
    if (history_.empty()) return SceneLabel::UNKNOWN;

    // Find the most frequent label
    const SceneLabel all_labels[] = {
        SceneLabel::SILENCE,
        SceneLabel::NOISE,
        SceneLabel::MUSIC,
        SceneLabel::SPEECH,
        SceneLabel::MIXED_SPEECH_NOISE,
        SceneLabel::UNKNOWN
    };

    SceneLabel best   = SceneLabel::UNKNOWN;
    int        best_n = 0;
    for (SceneLabel s : all_labels) {
        int n = count(s);
        if (n > best_n) { best_n = n; best = s; }
    }

    // Only declare dominance if threshold is met
    const double frac = static_cast<double>(best_n) / history_.size();
    if (frac >= cfg_.dominance_threshold) return best;

    return SceneLabel::UNKNOWN;  // no dominant scene
}

GateConfig AdaptiveController::adapt_gate(const GateConfig& base) const {
    if (!cfg_.enabled) return base;
    if (history_.empty()) return base;

    const SceneLabel dom = dominant_scene();

    GateConfig adapted = base;  // start from base thresholds

    switch (dom) {
        case SceneLabel::NOISE:
            // Tighten RMS floor: reject more quiet/noisy chunks in noise context.
            adapted.rms_min            = base.rms_min * cfg_.noise_rms_multiplier;
            adapted.rms_borderline_min = adapted.rms_min * 0.33;
            // Tighten flatness: reject moderately-flat chunks more aggressively.
            if (cfg_.noise_flatness_tight < base.spectral_flatness_max) {
                adapted.spectral_flatness_max  = cfg_.noise_flatness_tight;
                adapted.spectral_flatness_warn = cfg_.noise_flatness_tight * 0.85;
            }
            break;

        case SceneLabel::MUSIC:
            // No acoustic gate changes — ASR is skipped separately via skip_asr().
            break;

        case SceneLabel::SILENCE:
            // Already caught by gate at time-domain stage; no threshold change needed.
            break;

        case SceneLabel::SPEECH:
        case SceneLabel::MIXED_SPEECH_NOISE:
        case SceneLabel::UNKNOWN:
        default:
            // Use base thresholds unchanged.
            break;
    }

    return adapted;
}

bool AdaptiveController::skip_asr(SceneLabel scene) const {
    if (!cfg_.enabled) return false;
    if (cfg_.music_skip_asr   && scene == SceneLabel::MUSIC)   return true;
    if (cfg_.silence_skip_asr && scene == SceneLabel::SILENCE) return true;
    return false;
}

}  // namespace pipeline
