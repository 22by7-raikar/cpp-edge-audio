// File-level feature aggregation (Python parity rules) and admission policies.

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "edge_audio/audio/audio_io.h"
#include "edge_audio/quality/admission_policy.h"
#include "edge_audio/quality/file_features.h"
#include "edge_audio/quality/quality_model.h"
#include "edge_audio/report/run_logger.h"
#include "edge_audio/segmentation/fixed_chunker.h"

namespace {

/// Redirects std::cout into a string for the lifetime of the object.
class CoutCapture {
public:
    CoutCapture() : previous_(std::cout.rdbuf(buffer_.rdbuf())) {}
    ~CoutCapture() { std::cout.rdbuf(previous_); }
    std::string str() const { return buffer_.str(); }

private:
    std::ostringstream buffer_;
    std::streambuf* previous_;
};

bool near(float actual, double expected, double tolerance = 1e-6) {
    return std::fabs(static_cast<double>(actual) - expected) <= tolerance;
}

edge_audio::GateMetrics metrics_from(double first) {
    edge_audio::GateMetrics metrics;
    metrics.rms = first + 0.0;
    metrics.silence_ratio = first + 1.0;
    metrics.clipping_ratio = first + 2.0;
    metrics.zcr = first + 3.0;
    metrics.spectral_flatness = first + 4.0;
    metrics.spectral_centroid = first + 5.0;
    metrics.spectral_rolloff = first + 6.0;
    metrics.spectral_flux = first + 7.0;
    metrics.band_energy_low = first + 8.0;
    metrics.band_energy_mid = first + 9.0;
    metrics.band_energy_high = first + 10.0;
    metrics.active_frame_frac = first + 11.0;
    return metrics;
}

template <typename Function>
bool throws(Function function) {
    try {
        function();
    } catch (const std::exception&) {
        return true;
    }
    return false;
}

TEST(FileFeatures, ExactFeatureOrderSingleChunk) {
    edge_audio::QualityFeatureAggregator aggregator;
    aggregator.add(metrics_from(1.0), edge_audio::GateDecision::PASS);
    const auto features = aggregator.features();
    for (std::size_t index = 0; index < 12; ++index) {
        ASSERT_TRUE(near(features[index], 1.0 + index));
        ASSERT_TRUE(near(features[index + 12], 1.0 + index));
    }
    ASSERT_EQ(features[24], 1.0f);
    ASSERT_EQ(features[25], 0.0f);
    ASSERT_EQ(features[26], 0.0f);
}

TEST(FileFeatures, MultiChunkMeansMaximaAndFractions) {
    edge_audio::QualityFeatureAggregator aggregator;
    aggregator.add(metrics_from(1.0), edge_audio::GateDecision::PASS);
    aggregator.add(metrics_from(3.0), edge_audio::GateDecision::BORDERLINE);
    aggregator.add(metrics_from(5.0), edge_audio::GateDecision::FAIL);
    const auto features = aggregator.features();
    ASSERT_EQ(aggregator.chunk_count(), 3);
    for (std::size_t index = 0; index < 12; ++index) {
        ASSERT_TRUE(near(features[index], 3.0 + index));
        ASSERT_TRUE(near(features[index + 12], 5.0 + index));
    }
    ASSERT_TRUE(near(features[24], 1.0 / 3.0));
    ASSERT_TRUE(near(features[25], 1.0 / 3.0));
    ASSERT_TRUE(near(features[26], 1.0 / 3.0));
}

TEST(FileFeatures, PythonSixDecimalRecordParity) {
    edge_audio::QualityFeatureAggregator aggregator;
    auto first = metrics_from(0.0);
    auto second = metrics_from(0.0);
    first.rms = 0.1234566;
    second.rms = 0.1234554;
    aggregator.add(first, edge_audio::GateDecision::PASS);
    aggregator.add(second, edge_audio::GateDecision::FAIL);
    const auto features = aggregator.features();
    ASSERT_TRUE(near(features[0], 0.123456, 1e-7));
    ASSERT_TRUE(near(features[12], 0.123457, 1e-7));
}

TEST(FileFeatures, FinalPartialChunkIsAggregated) {
    edge_audio::AudioBuffer audio;
    audio.sample_rate = 10;
    audio.channels = 1;
    audio.samples.resize(110, 0.1f);
    edge_audio::ChunkerConfig config;
    config.chunk_ms = 5000;
    config.hop_ms = 0;
    const auto chunks = edge_audio::chunk_audio(audio, config);
    ASSERT_EQ(chunks.size(), 3);
    ASSERT_TRUE(std::fabs(chunks.back().duration_sec() - 1.0) < 1e-12);

    edge_audio::QualityFeatureAggregator aggregator;
    for (const auto& chunk : chunks) {
        aggregator.add(metrics_from(static_cast<double>(chunk.index + 1)),
                       edge_audio::GateDecision::PASS);
    }
    const auto features = aggregator.features();
    ASSERT_EQ(aggregator.chunk_count(), 3);
    ASSERT_TRUE(near(features[0], 2.0));
    ASSERT_TRUE(near(features[12], 3.0));
}

TEST(FileFeatures, EmptyAndReset) {
    edge_audio::QualityFeatureAggregator aggregator;
    ASSERT_TRUE(aggregator.empty());
    ASSERT_TRUE(throws([&] { (void)aggregator.features(); }));
    aggregator.add(metrics_from(2.0), edge_audio::GateDecision::FAIL);
    ASSERT_FALSE(aggregator.empty());
    aggregator.reset();
    ASSERT_TRUE(aggregator.empty());
    ASSERT_EQ(aggregator.chunk_count(), 0);
    ASSERT_TRUE(throws([&] { (void)aggregator.features(); }));
    aggregator.add(metrics_from(4.0), edge_audio::GateDecision::BORDERLINE);
    const auto features = aggregator.features();
    ASSERT_TRUE(near(features[0], 4.0));
    ASSERT_EQ(features[25], 1.0f);
}

TEST(FileFeatures, NonFiniteMetricRejected) {
    edge_audio::QualityFeatureAggregator aggregator;
    auto metrics = metrics_from(0.0);
    metrics.spectral_flux = std::numeric_limits<double>::infinity();
    ASSERT_TRUE(throws([&] { aggregator.add(metrics, edge_audio::GateDecision::PASS); }));
    ASSERT_TRUE(aggregator.empty());
    aggregator.add(metrics_from(2.0), edge_audio::GateDecision::PASS);
    ASSERT_TRUE(near(aggregator.features()[0], 2.0));
}

TEST(FileFeatures, DefaultRuleBehaviorUnchanged) {
    edge_audio::QualityPrediction learned;
    learned.should_transcribe = true;
    const auto reject =
        edge_audio::decide_quality_admission(edge_audio::QualityPolicy::RULE, false, &learned);
    const auto admit =
        edge_audio::decide_quality_admission(edge_audio::QualityPolicy::RULE, true, nullptr);
    ASSERT_FALSE(reject.final_should_transcribe);
    ASSERT_EQ(reject.rejection_reason, "rule_rejected");
    ASSERT_TRUE(admit.final_should_transcribe);
    ASSERT_TRUE(admit.rejection_reason.empty());
}

TEST(FileFeatures, LearnedThresholdBoundaryAndAlternate) {
    std::array<float, edge_audio::kQualityFeatureCount> features{};
    const auto reference = edge_audio::predict_quality(features);
    const auto boundary = edge_audio::predict_quality(features, reference.probability);
    ASSERT_TRUE(boundary.should_transcribe);
    const double above = std::nextafter(reference.probability, 1.0);
    if (above <= 1.0) {
        ASSERT_FALSE(edge_audio::predict_quality(features, above).should_transcribe);
    }
    ASSERT_TRUE(edge_audio::predict_quality(features, 0.0).should_transcribe);
}

TEST(FileFeatures, LearnedAcceptAndReject) {
    edge_audio::QualityPrediction accept;
    accept.should_transcribe = true;
    edge_audio::QualityPrediction reject;
    reject.should_transcribe = false;
    ASSERT_TRUE(
        edge_audio::decide_quality_admission(edge_audio::QualityPolicy::LEARNED, false, &accept)
            .final_should_transcribe);
    const auto rejected =
        edge_audio::decide_quality_admission(edge_audio::QualityPolicy::LEARNED, true, &reject);
    ASSERT_FALSE(rejected.final_should_transcribe);
    ASSERT_EQ(rejected.rejection_reason, "learned_below_threshold");
}

TEST(FileFeatures, HybridTruthTable) {
    edge_audio::QualityPrediction learned_accept;
    learned_accept.should_transcribe = true;
    edge_audio::QualityPrediction learned_reject;
    learned_reject.should_transcribe = false;
    ASSERT_TRUE(edge_audio::decide_quality_admission(edge_audio::QualityPolicy::HYBRID, true,
                                                     &learned_accept)
                    .final_should_transcribe);
    ASSERT_FALSE(edge_audio::decide_quality_admission(edge_audio::QualityPolicy::HYBRID, true,
                                                      &learned_reject)
                     .final_should_transcribe);
    ASSERT_FALSE(edge_audio::decide_quality_admission(edge_audio::QualityPolicy::HYBRID, false,
                                                      &learned_accept)
                     .final_should_transcribe);
    ASSERT_FALSE(edge_audio::decide_quality_admission(edge_audio::QualityPolicy::HYBRID, false,
                                                      &learned_reject)
                     .final_should_transcribe);
}

TEST(FileFeatures, PolicyAndThresholdValidation) {
    ASSERT_EQ(edge_audio::parse_quality_policy("rule"), edge_audio::QualityPolicy::RULE);
    ASSERT_EQ(edge_audio::parse_quality_policy("learned"), edge_audio::QualityPolicy::LEARNED);
    ASSERT_EQ(edge_audio::parse_quality_policy("hybrid"), edge_audio::QualityPolicy::HYBRID);
    ASSERT_TRUE(throws([] { (void)edge_audio::parse_quality_policy("other"); }));
    ASSERT_EQ(edge_audio::parse_quality_threshold("0"), 0.0);
    ASSERT_EQ(edge_audio::parse_quality_threshold("0.3"), 0.3);
    ASSERT_EQ(edge_audio::parse_quality_threshold("1"), 1.0);
    ASSERT_TRUE(throws([] { (void)edge_audio::parse_quality_threshold("-0.01"); }));
    ASSERT_TRUE(throws([] { (void)edge_audio::parse_quality_threshold("1.01"); }));
    ASSERT_TRUE(throws([] { (void)edge_audio::parse_quality_threshold("nan"); }));
    ASSERT_TRUE(throws([] { (void)edge_audio::parse_quality_threshold("0.3x"); }));
}

TEST(FileFeatures, AsrCalledOnceForAcceptedLearnedFile) {
    const std::vector<float> audio(16000, 0.1f);
    int calls = 0;
    bool pointer_matched = false;
    bool count_matched = false;
    const auto execution = edge_audio::transcribe_if_admitted(
        true, true, audio.data(), audio.size(), [&](const float* samples, int sample_count) {
            ++calls;
            pointer_matched = samples == audio.data();
            count_matched = sample_count == static_cast<int>(audio.size());
            edge_audio::AsrResult result;
            result.ok = true;
            return result;
        });
    ASSERT_TRUE(execution.ran);
    ASSERT_EQ(calls, 1);
    ASSERT_TRUE(pointer_matched);
    ASSERT_TRUE(count_matched);
}

TEST(FileFeatures, AsrNotCalledForRejectedLearnedFile) {
    const std::vector<float> audio(16000, 0.1f);
    int calls = 0;
    const auto execution = edge_audio::transcribe_if_admitted(false, true, audio.data(),
                                                              audio.size(), [&](const float*, int) {
                                                                  ++calls;
                                                                  return edge_audio::AsrResult{};
                                                              });
    ASSERT_FALSE(execution.ran);
    ASSERT_EQ(calls, 0);
}

TEST(FileFeatures, QualitySummaryLogFieldsAndDefaultFeaturePrivacy) {
    const std::string json_path = ::testing::TempDir() + "edge_audio_quality_summary.json";
    CoutCapture captured;
    edge_audio::RunLogger logger;
    logger.set_json_path(json_path);
    edge_audio::RunConfig config;
    config.input_path = "test.wav";
    config.quality_policy = "learned";
    config.quality_threshold = 0.3;
    logger.log_run_start(config);

    edge_audio::QualitySummaryRecord summary;
    summary.policy = "learned";
    summary.chunk_count = 2;
    summary.features_available = true;
    summary.learned_evaluated = true;
    summary.learned_raw_score = -0.25;
    summary.learned_probability = 0.437823499;
    summary.learned_inference_us = 0.8;
    summary.learned_decision = true;
    summary.rule_summary = false;
    summary.final_admission = true;
    summary.asr_ran = true;
    logger.log_quality_summary(summary);
    ASSERT_TRUE(logger.log_run_end(2, 1, 1, 0, 10.0, 100.0));

    auto read_file = [](const std::string& path) {
        std::ifstream input(path);
        return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
    };
    const std::string tsv = captured.str();
    const std::string json = read_file(json_path);
    ASSERT_NE(tsv.find("event=quality_summary"), std::string::npos);
    ASSERT_NE(tsv.find("policy=learned"), std::string::npos);
    ASSERT_NE(tsv.find("chunk_count=2"), std::string::npos);
    ASSERT_NE(tsv.find("schema_version=quality-file-features-v1"), std::string::npos);
    ASSERT_NE(tsv.find("learned_raw_score=-0.25"), std::string::npos);
    ASSERT_NE(tsv.find("learned_probability="), std::string::npos);
    ASSERT_NE(tsv.find("rule_summary=reject"), std::string::npos);
    ASSERT_NE(tsv.find("final_admission=admit"), std::string::npos);
    ASSERT_NE(tsv.find("asr_ran=1"), std::string::npos);
    ASSERT_EQ(tsv.find("quality_features="), std::string::npos);
    ASSERT_NE(json.find("\"quality\""), std::string::npos);
    ASSERT_NE(json.find("\"learned_raw_score\""), std::string::npos);
    ASSERT_EQ(json.find("\"features\""), std::string::npos);
}

}  // namespace
