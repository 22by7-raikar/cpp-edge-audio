#pragma once

#include <array>
#include <cstddef>

#include "audio/audio_io.h"
#include "gate/quality_model.h"
#include "gate/quality_policy.h"
#include "scene/scene.h"

namespace pipeline {

struct CompletedUtteranceAnalysis {
    AudioBuffer audio_16khz;
    std::array<float, kQualityFeatureCount> quality_features{};
    std::size_t analysis_chunks = 0;
    bool rule_summary = false;
    bool learned_evaluated = false;
    QualityPrediction quality_prediction;
    QualityAdmission admission;
    SceneResult scene;
    double quality_us = 0.0;
};

// Runs the frozen file-equivalent quality path after an utterance boundary.
// Learned inference, when selected, occurs exactly once after all five-second
// chunk metrics have been aggregated.
class CompletedUtteranceAnalyzer {
public:
    CompletedUtteranceAnalysis analyze(
        AudioBuffer utterance,
        QualityPolicy policy,
        double threshold = kQualityDefaultThreshold) const;
};

}  // namespace pipeline
