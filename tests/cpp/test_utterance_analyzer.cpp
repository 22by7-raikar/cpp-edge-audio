// UtteranceAnalyzer: file-equivalent chunking, resampling and policy semantics.

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "edge_audio/quality/utterance_analyzer.h"

namespace {

edge_audio::AudioBuffer sine_audio(int sample_rate, double seconds) {
    edge_audio::AudioBuffer audio;
    audio.sample_rate = sample_rate;
    audio.channels = 1;
    const int count = static_cast<int>(sample_rate * seconds);
    audio.samples.resize(static_cast<std::size_t>(count));
    for (int index = 0; index < count; ++index) {
        audio.samples[static_cast<std::size_t>(index)] =
            0.2f * std::sin(2.0 * 3.14159265358979323846 * 1000.0 * index / sample_rate);
    }
    return audio;
}

TEST(UtteranceAnalyzer, CompletedUtteranceUsesFrozenChunkAggregation) {
    edge_audio::UtteranceAnalyzer analyzer;
    const auto result = analyzer.analyze(sine_audio(16000, 5.25), edge_audio::QualityPolicy::RULE);

    ASSERT_EQ(result.audio_16khz.sample_rate, 16000);
    ASSERT_EQ(result.analysis_chunks, 2);
    ASSERT_EQ(result.quality_features.size(), edge_audio::kQualityFeatureCount);
    for (float value : result.quality_features) {
        ASSERT_TRUE(std::isfinite(static_cast<double>(value)));
    }
    ASSERT_TRUE(result.rule_summary);
    ASSERT_TRUE(result.admission.final_should_transcribe);
    ASSERT_FALSE(result.learned_evaluated);
    ASSERT_GE(result.quality_us, 0.0);
}

TEST(UtteranceAnalyzer, Non16khzUsesExistingResample) {
    edge_audio::UtteranceAnalyzer analyzer;
    const edge_audio::AudioBuffer input = sine_audio(8000, 1.0);
    const edge_audio::AudioBuffer expected = edge_audio::resample(input, 16000);
    const auto result = analyzer.analyze(input, edge_audio::QualityPolicy::RULE);

    ASSERT_EQ(result.audio_16khz.sample_rate, 16000);
    ASSERT_EQ(result.audio_16khz.samples.size(), expected.samples.size());
    ASSERT_EQ(result.audio_16khz.samples, expected.samples);
    ASSERT_EQ(result.analysis_chunks, 1);
}

TEST(UtteranceAnalyzer, LearnedRejectsZeroQualityUtterance) {
    edge_audio::AudioBuffer audio;
    audio.sample_rate = 16000;
    audio.channels = 1;
    audio.samples.assign(16000, 0.0f);

    edge_audio::UtteranceAnalyzer analyzer;
    const auto result = analyzer.analyze(std::move(audio), edge_audio::QualityPolicy::LEARNED);

    ASSERT_TRUE(result.learned_evaluated);
    ASSERT_TRUE(result.quality_prediction.probability < edge_audio::kQualityDefaultThreshold);
    ASSERT_FALSE(result.admission.final_should_transcribe);
    ASSERT_EQ(result.admission.rejection_reason, "learned_below_threshold");
}

TEST(UtteranceAnalyzer, RuleAndHybridUseExistingPolicySemantics) {
    edge_audio::UtteranceAnalyzer analyzer;

    const auto rule = analyzer.analyze(sine_audio(16000, 1.0), edge_audio::QualityPolicy::RULE);
    const auto expected_rule = edge_audio::decide_quality_admission(edge_audio::QualityPolicy::RULE,
                                                                    rule.rule_summary, nullptr);
    ASSERT_EQ(rule.admission.final_should_transcribe, expected_rule.final_should_transcribe);
    ASSERT_EQ(rule.admission.rejection_reason, expected_rule.rejection_reason);

    const auto hybrid = analyzer.analyze(sine_audio(16000, 1.0), edge_audio::QualityPolicy::HYBRID);
    const auto expected_hybrid = edge_audio::decide_quality_admission(
        edge_audio::QualityPolicy::HYBRID, hybrid.rule_summary, &hybrid.quality_prediction);
    ASSERT_TRUE(hybrid.learned_evaluated);
    ASSERT_EQ(hybrid.admission.final_should_transcribe, expected_hybrid.final_should_transcribe);
    ASSERT_EQ(hybrid.admission.rejection_reason, expected_hybrid.rejection_reason);
}

TEST(UtteranceAnalyzer, ThresholdControlsLearnedDecision) {
    const edge_audio::UtteranceAnalyzer analyzer;
    const auto at_zero =
        analyzer.analyze(sine_audio(16000, 1.0), edge_audio::QualityPolicy::LEARNED, 0.0);
    EXPECT_TRUE(at_zero.admission.final_should_transcribe);
    const auto at_one =
        analyzer.analyze(sine_audio(16000, 1.0), edge_audio::QualityPolicy::LEARNED, 1.0);
    EXPECT_EQ(at_one.quality_prediction.probability, at_zero.quality_prediction.probability);
    EXPECT_EQ(at_one.admission.final_should_transcribe,
              at_one.quality_prediction.probability >= 1.0);
}

TEST(UtteranceAnalyzer, RejectsInvalidThreshold) {
    const edge_audio::UtteranceAnalyzer analyzer;
    EXPECT_THROW(
        (void)analyzer.analyze(sine_audio(16000, 1.0), edge_audio::QualityPolicy::LEARNED, 1.5),
        std::invalid_argument);
}

}  // namespace
