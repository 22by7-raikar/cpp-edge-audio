/// @file
/// @brief Quality admission for one completed utterance.
/// @ingroup quality

#pragma once

#include <array>
#include <cstddef>

#include "edge_audio/audio/audio_io.h"
#include "edge_audio/quality/admission_policy.h"
#include "edge_audio/quality/quality_model.h"
#include "edge_audio/quality/scene_classifier.h"

namespace edge_audio {

/// @addtogroup quality
/// @{

/// @brief Everything known about an utterance after quality analysis.
struct UtteranceAnalysis {
    AudioBuffer audio_16khz;  ///< The utterance at 16 kHz, ready for ASR.
    std::array<float, kQualityFeatureCount> quality_features{};  ///< Learned-model input.
    std::size_t analysis_chunks = 0;                             ///< 5 s windows analysed.
    bool rule_summary = false;             ///< True if any window passed the rule gate.
    bool learned_evaluated = false;        ///< True under LEARNED / HYBRID.
    QualityPrediction quality_prediction;  ///< Valid when #learned_evaluated.
    QualityAdmission admission;            ///< Final verdict.
    SceneResult scene;                     ///< Highest-confidence scene across windows.
    double quality_us = 0.0;               ///< Wall-clock analysis time, microseconds.
};

/// @brief Runs the offline file-level quality path on one utterance, so that
/// streaming admission makes exactly the decision the model was evaluated on.
///
/// Resample to 16 kHz → non-overlapping 5 s windows → rule gate and scene per
/// window → 27-feature vector → (if learned/hybrid) one model evaluation →
/// policy verdict. The adaptive controller is not applied: utterances are
/// independent and short.
class UtteranceAnalyzer {
public:
    /// @throws std::invalid_argument if @p utterance is empty or the threshold is invalid.
    UtteranceAnalysis analyze(AudioBuffer utterance, QualityPolicy policy,
                              double threshold = kQualityDefaultThreshold) const;
};

/// @}

}  // namespace edge_audio
