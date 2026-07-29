#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "pipeline/completed_utterance_analyzer.h"

namespace {

int passed = 0;
int failed = 0;

#define CHECK(condition)                                                   \
    do {                                                                   \
        if (!(condition)) {                                                \
            std::cerr << "FAIL: " << __FILE__ << ":" << __LINE__         \
                      << "  " << #condition << "\n";                     \
            return false;                                                  \
        }                                                                  \
    } while (0)

bool run(const char* name, bool (*test)()) {
    const bool ok = test();
    std::cout << "  " << (ok ? "PASS" : "FAIL") << "  " << name << "\n";
    ok ? ++passed : ++failed;
    return ok;
}

pipeline::AudioBuffer sine_audio(int sample_rate, double seconds) {
    pipeline::AudioBuffer audio;
    audio.sample_rate = sample_rate;
    audio.channels = 1;
    const int count = static_cast<int>(sample_rate * seconds);
    audio.samples.resize(static_cast<std::size_t>(count));
    for (int index = 0; index < count; ++index) {
        audio.samples[static_cast<std::size_t>(index)] =
            0.2f * std::sin(
                2.0 * 3.14159265358979323846 * 1000.0 * index / sample_rate);
    }
    return audio;
}

bool test_completed_utterance_uses_frozen_chunk_aggregation() {
    pipeline::CompletedUtteranceAnalyzer analyzer;
    const auto result = analyzer.analyze(
        sine_audio(16000, 5.25),
        pipeline::QualityPolicy::RULE);

    CHECK(result.audio_16khz.sample_rate == 16000);
    CHECK(result.analysis_chunks == 2);
    CHECK(result.quality_features.size() == pipeline::kQualityFeatureCount);
    for (float value : result.quality_features) {
        CHECK(std::isfinite(static_cast<double>(value)));
    }
    CHECK(result.rule_summary);
    CHECK(result.admission.final_should_transcribe);
    CHECK(!result.learned_evaluated);
    CHECK(result.quality_us >= 0.0);
    return true;
}

bool test_non_16khz_uses_existing_resample() {
    pipeline::CompletedUtteranceAnalyzer analyzer;
    const pipeline::AudioBuffer input = sine_audio(8000, 1.0);
    const pipeline::AudioBuffer expected = pipeline::resample(input, 16000);
    const auto result = analyzer.analyze(
        input, pipeline::QualityPolicy::RULE);

    CHECK(result.audio_16khz.sample_rate == 16000);
    CHECK(result.audio_16khz.samples.size() == expected.samples.size());
    CHECK(result.audio_16khz.samples == expected.samples);
    CHECK(result.analysis_chunks == 1);
    return true;
}

bool test_learned_rejects_zero_quality_utterance() {
    pipeline::AudioBuffer audio;
    audio.sample_rate = 16000;
    audio.channels = 1;
    audio.samples.assign(16000, 0.0f);

    pipeline::CompletedUtteranceAnalyzer analyzer;
    const auto result = analyzer.analyze(
        std::move(audio), pipeline::QualityPolicy::LEARNED);

    CHECK(result.learned_evaluated);
    CHECK(result.quality_prediction.probability <
          pipeline::kQualityDefaultThreshold);
    CHECK(!result.admission.final_should_transcribe);
    CHECK(result.admission.rejection_reason == "learned_below_threshold");
    return true;
}

bool test_rule_and_hybrid_use_existing_policy_semantics() {
    pipeline::CompletedUtteranceAnalyzer analyzer;

    const auto rule = analyzer.analyze(
        sine_audio(16000, 1.0), pipeline::QualityPolicy::RULE);
    const auto expected_rule = pipeline::decide_quality_admission(
        pipeline::QualityPolicy::RULE, rule.rule_summary, nullptr);
    CHECK(rule.admission.final_should_transcribe ==
          expected_rule.final_should_transcribe);
    CHECK(rule.admission.rejection_reason == expected_rule.rejection_reason);

    const auto hybrid = analyzer.analyze(
        sine_audio(16000, 1.0), pipeline::QualityPolicy::HYBRID);
    const auto expected_hybrid = pipeline::decide_quality_admission(
        pipeline::QualityPolicy::HYBRID,
        hybrid.rule_summary,
        &hybrid.quality_prediction);
    CHECK(hybrid.learned_evaluated);
    CHECK(hybrid.admission.final_should_transcribe ==
          expected_hybrid.final_should_transcribe);
    CHECK(hybrid.admission.rejection_reason ==
          expected_hybrid.rejection_reason);
    return true;
}

bool test_alternate_threshold_is_rejected() {
    pipeline::CompletedUtteranceAnalyzer analyzer;
    try {
        (void)analyzer.analyze(
            sine_audio(16000, 1.0),
            pipeline::QualityPolicy::LEARNED,
            0.4);
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

}  // namespace

int main() {
    std::cout << "Running completed-utterance analyzer tests...\n\n";
    run("frozen_chunk_aggregation",
        test_completed_utterance_uses_frozen_chunk_aggregation);
    run("non_16khz_existing_resample",
        test_non_16khz_uses_existing_resample);
    run("learned_reject_zero_quality",
        test_learned_rejects_zero_quality_utterance);
    run("rule_hybrid_existing_semantics",
        test_rule_and_hybrid_use_existing_policy_semantics);
    run("alternate_threshold_rejected",
        test_alternate_threshold_is_rejected);
    std::cout << "\n" << passed << " passed, " << failed << " failed\n";
    return failed == 0 ? 0 : 1;
}
