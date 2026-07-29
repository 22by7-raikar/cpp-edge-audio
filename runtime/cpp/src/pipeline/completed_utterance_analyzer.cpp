#include "pipeline/completed_utterance_analyzer.h"

#include <chrono>
#include <stdexcept>
#include <utility>

#include "chunker/chunker.h"
#include "gate/gate.h"
#include "gate/quality_aggregator.h"

namespace pipeline {

CompletedUtteranceAnalysis CompletedUtteranceAnalyzer::analyze(
    AudioBuffer utterance,
    QualityPolicy policy,
    double threshold) const {
    if (utterance.samples.empty() || utterance.sample_rate <= 0) {
        throw std::invalid_argument("completed utterance has no valid audio");
    }
    if (threshold != kQualityDefaultThreshold) {
        throw std::invalid_argument(
            "robot hearing requires the frozen quality threshold 0.3");
    }

    const auto started = std::chrono::steady_clock::now();

    CompletedUtteranceAnalysis result;
    result.audio_16khz = utterance.sample_rate == 16000
        ? std::move(utterance)
        : resample(utterance, 16000);

    ChunkerConfig chunk_config;
    chunk_config.chunk_ms = 5000;
    chunk_config.hop_ms = 0;
    const auto chunks = chunk_audio(result.audio_16khz, chunk_config);
    if (chunks.empty()) {
        throw std::runtime_error(
            "completed utterance produced no quality-analysis chunks");
    }

    GateConfig gate_config;
    SceneConfig scene_config;
    QualityFeatureAggregator aggregator;
    bool have_scene = false;

    for (const auto& chunk : chunks) {
        const GateResult gate = evaluate_chunk(chunk, gate_config);
        aggregator.add(gate.metrics, gate.decision);
        if (gate.decision == GateDecision::PASS ||
            gate.decision == GateDecision::BORDERLINE) {
            result.rule_summary = true;
        }

        const SceneResult scene = classify(gate.metrics, scene_config);
        if (!have_scene || scene.confidence > result.scene.confidence) {
            result.scene = scene;
            have_scene = true;
        }
    }

    result.analysis_chunks = aggregator.chunk_count();
    result.quality_features = aggregator.features();

    const QualityPrediction* prediction = nullptr;
    if (policy != QualityPolicy::RULE) {
        result.quality_prediction =
            predict_quality(result.quality_features, threshold);
        result.learned_evaluated = true;
        prediction = &result.quality_prediction;
    }
    result.admission =
        decide_quality_admission(policy, result.rule_summary, prediction);

    const auto finished = std::chrono::steady_clock::now();
    result.quality_us =
        std::chrono::duration<double, std::micro>(finished - started).count();
    return result;
}

}  // namespace pipeline
