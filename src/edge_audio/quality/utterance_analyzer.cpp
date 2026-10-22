#include "edge_audio/quality/utterance_analyzer.h"

#include <chrono>
#include <cmath>
#include <stdexcept>
#include <utility>

#include "edge_audio/quality/file_features.h"
#include "edge_audio/quality/rule_gate.h"
#include "edge_audio/segmentation/fixed_chunker.h"

namespace edge_audio {
namespace {

constexpr int kAnalysisRate = 16000;    // ASR and model input rate
constexpr int kAnalysisChunkMs = 5000;  // training-time window, non-overlapping

}  // namespace

UtteranceAnalysis UtteranceAnalyzer::analyze(AudioBuffer utterance, QualityPolicy policy,
                                             double threshold) const {
    if (utterance.samples.empty() || utterance.sample_rate <= 0) {
        throw std::invalid_argument("utterance has no audio");
    }
    if (!std::isfinite(threshold) || threshold < 0.0 || threshold > 1.0) {
        throw std::invalid_argument("quality threshold must be in [0, 1]");
    }

    const auto started = std::chrono::steady_clock::now();

    UtteranceAnalysis result;
    result.audio_16khz = utterance.sample_rate == kAnalysisRate
                             ? std::move(utterance)
                             : resample(utterance, kAnalysisRate);

    ChunkerConfig chunk_config;
    chunk_config.chunk_ms = kAnalysisChunkMs;
    const auto chunks = chunk_audio(result.audio_16khz, chunk_config);
    if (chunks.empty()) {
        throw std::runtime_error("utterance produced no analysis chunks");
    }

    const GateConfig gate_config;
    const SceneConfig scene_config;
    QualityFeatureAggregator aggregator;
    bool have_scene = false;
    for (const auto& chunk : chunks) {
        const GateResult gate = evaluate_chunk(chunk, gate_config);
        aggregator.add(gate.metrics, gate.decision);
        if (gate.decision != GateDecision::FAIL) {
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
        result.quality_prediction = predict_quality(result.quality_features, threshold);
        result.learned_evaluated = true;
        prediction = &result.quality_prediction;
    }
    result.admission = decide_quality_admission(policy, result.rule_summary, prediction);

    result.quality_us =
        std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - started)
            .count();
    return result;
}

}  // namespace edge_audio
