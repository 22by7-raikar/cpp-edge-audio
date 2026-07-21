#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "audio/audio_io.h"
#include "chunker/chunker.h"
#include "gate/quality_aggregator.h"
#include "gate/quality_model.h"
#include "gate/quality_policy.h"
#include "logging/logger.h"

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

bool near(float actual, double expected, double tolerance = 1e-6) {
    return std::fabs(static_cast<double>(actual) - expected) <= tolerance;
}

pipeline::GateMetrics metrics_from(double first) {
    pipeline::GateMetrics metrics;
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

bool test_exact_feature_order_single_chunk() {
    pipeline::QualityFeatureAggregator aggregator;
    aggregator.add(metrics_from(1.0), pipeline::GateDecision::PASS);
    const auto features = aggregator.features();
    for (std::size_t index = 0; index < 12; ++index) {
        CHECK(near(features[index], 1.0 + index));
        CHECK(near(features[index + 12], 1.0 + index));
    }
    CHECK(features[24] == 1.0f);
    CHECK(features[25] == 0.0f);
    CHECK(features[26] == 0.0f);
    return true;
}

bool test_multi_chunk_means_maxima_and_fractions() {
    pipeline::QualityFeatureAggregator aggregator;
    aggregator.add(metrics_from(1.0), pipeline::GateDecision::PASS);
    aggregator.add(metrics_from(3.0), pipeline::GateDecision::BORDERLINE);
    aggregator.add(metrics_from(5.0), pipeline::GateDecision::FAIL);
    const auto features = aggregator.features();
    CHECK(aggregator.chunk_count() == 3);
    for (std::size_t index = 0; index < 12; ++index) {
        CHECK(near(features[index], 3.0 + index));
        CHECK(near(features[index + 12], 5.0 + index));
    }
    CHECK(near(features[24], 1.0 / 3.0));
    CHECK(near(features[25], 1.0 / 3.0));
    CHECK(near(features[26], 1.0 / 3.0));
    return true;
}

bool test_python_six_decimal_record_parity() {
    pipeline::QualityFeatureAggregator aggregator;
    auto first = metrics_from(0.0);
    auto second = metrics_from(0.0);
    first.rms = 0.1234566;
    second.rms = 0.1234554;
    aggregator.add(first, pipeline::GateDecision::PASS);
    aggregator.add(second, pipeline::GateDecision::FAIL);
    const auto features = aggregator.features();
    CHECK(near(features[0], 0.123456, 1e-7));
    CHECK(near(features[12], 0.123457, 1e-7));
    return true;
}

bool test_final_partial_chunk_is_aggregated() {
    pipeline::AudioBuffer audio;
    audio.sample_rate = 10;
    audio.channels = 1;
    audio.samples.resize(110, 0.1f);
    pipeline::ChunkerConfig config;
    config.chunk_ms = 5000;
    config.hop_ms = 0;
    const auto chunks = pipeline::chunk_audio(audio, config);
    CHECK(chunks.size() == 3);
    CHECK(std::fabs(chunks.back().duration_sec() - 1.0) < 1e-12);

    pipeline::QualityFeatureAggregator aggregator;
    for (const auto& chunk : chunks) {
        aggregator.add(
            metrics_from(static_cast<double>(chunk.index + 1)),
            pipeline::GateDecision::PASS);
    }
    const auto features = aggregator.features();
    CHECK(aggregator.chunk_count() == 3);
    CHECK(near(features[0], 2.0));
    CHECK(near(features[12], 3.0));
    return true;
}

bool test_empty_and_reset() {
    pipeline::QualityFeatureAggregator aggregator;
    CHECK(aggregator.empty());
    CHECK(throws([&] { (void)aggregator.features(); }));
    aggregator.add(metrics_from(2.0), pipeline::GateDecision::FAIL);
    CHECK(!aggregator.empty());
    aggregator.reset();
    CHECK(aggregator.empty());
    CHECK(aggregator.chunk_count() == 0);
    CHECK(throws([&] { (void)aggregator.features(); }));
    aggregator.add(metrics_from(4.0), pipeline::GateDecision::BORDERLINE);
    const auto features = aggregator.features();
    CHECK(near(features[0], 4.0));
    CHECK(features[25] == 1.0f);
    return true;
}

bool test_non_finite_metric_rejected() {
    pipeline::QualityFeatureAggregator aggregator;
    auto metrics = metrics_from(0.0);
    metrics.spectral_flux = std::numeric_limits<double>::infinity();
    CHECK(throws([&] {
        aggregator.add(metrics, pipeline::GateDecision::PASS);
    }));
    CHECK(aggregator.empty());
    aggregator.add(metrics_from(2.0), pipeline::GateDecision::PASS);
    CHECK(near(aggregator.features()[0], 2.0));
    return true;
}

bool test_default_rule_behavior_unchanged() {
    pipeline::QualityPrediction learned;
    learned.should_transcribe = true;
    const auto reject = pipeline::decide_quality_admission(
        pipeline::QualityPolicy::RULE, false, &learned);
    const auto admit = pipeline::decide_quality_admission(
        pipeline::QualityPolicy::RULE, true, nullptr);
    CHECK(!reject.final_should_transcribe);
    CHECK(reject.rejection_reason == "rule_rejected");
    CHECK(admit.final_should_transcribe);
    CHECK(admit.rejection_reason.empty());
    return true;
}

bool test_learned_threshold_boundary_and_alternate() {
    std::array<float, pipeline::kQualityFeatureCount> features{};
    const auto reference = pipeline::predict_quality(features);
    const auto boundary = pipeline::predict_quality(
        features, reference.probability);
    CHECK(boundary.should_transcribe);
    const double above = std::nextafter(reference.probability, 1.0);
    if (above <= 1.0) {
        CHECK(!pipeline::predict_quality(features, above).should_transcribe);
    }
    CHECK(pipeline::predict_quality(features, 0.0).should_transcribe);
    return true;
}

bool test_learned_accept_and_reject() {
    pipeline::QualityPrediction accept;
    accept.should_transcribe = true;
    pipeline::QualityPrediction reject;
    reject.should_transcribe = false;
    CHECK(pipeline::decide_quality_admission(
              pipeline::QualityPolicy::LEARNED, false, &accept)
              .final_should_transcribe);
    const auto rejected = pipeline::decide_quality_admission(
        pipeline::QualityPolicy::LEARNED, true, &reject);
    CHECK(!rejected.final_should_transcribe);
    CHECK(rejected.rejection_reason == "learned_below_threshold");
    return true;
}

bool test_hybrid_truth_table() {
    pipeline::QualityPrediction learned_accept;
    learned_accept.should_transcribe = true;
    pipeline::QualityPrediction learned_reject;
    learned_reject.should_transcribe = false;
    CHECK(pipeline::decide_quality_admission(
              pipeline::QualityPolicy::HYBRID, true, &learned_accept)
              .final_should_transcribe);
    CHECK(!pipeline::decide_quality_admission(
               pipeline::QualityPolicy::HYBRID, true, &learned_reject)
               .final_should_transcribe);
    CHECK(!pipeline::decide_quality_admission(
               pipeline::QualityPolicy::HYBRID, false, &learned_accept)
               .final_should_transcribe);
    CHECK(!pipeline::decide_quality_admission(
               pipeline::QualityPolicy::HYBRID, false, &learned_reject)
               .final_should_transcribe);
    return true;
}

bool test_policy_and_threshold_validation() {
    CHECK(pipeline::parse_quality_policy("rule") == pipeline::QualityPolicy::RULE);
    CHECK(pipeline::parse_quality_policy("learned") == pipeline::QualityPolicy::LEARNED);
    CHECK(pipeline::parse_quality_policy("hybrid") == pipeline::QualityPolicy::HYBRID);
    CHECK(throws([] { (void)pipeline::parse_quality_policy("other"); }));
    CHECK(pipeline::parse_quality_threshold("0") == 0.0);
    CHECK(pipeline::parse_quality_threshold("0.3") == 0.3);
    CHECK(pipeline::parse_quality_threshold("1") == 1.0);
    CHECK(throws([] { (void)pipeline::parse_quality_threshold("-0.01"); }));
    CHECK(throws([] { (void)pipeline::parse_quality_threshold("1.01"); }));
    CHECK(throws([] { (void)pipeline::parse_quality_threshold("nan"); }));
    CHECK(throws([] { (void)pipeline::parse_quality_threshold("0.3x"); }));
    return true;
}

bool test_asr_called_once_for_accepted_learned_file() {
    const std::vector<float> audio(16000, 0.1f);
    int calls = 0;
    bool pointer_matched = false;
    bool count_matched = false;
    const auto execution = pipeline::transcribe_admitted_file_once(
        true,
        true,
        audio.data(),
        audio.size(),
        [&](const float* samples, int sample_count) {
            ++calls;
            pointer_matched = samples == audio.data();
            count_matched = sample_count == static_cast<int>(audio.size());
            pipeline::AsrResult result;
            result.ok = true;
            return result;
        });
    CHECK(execution.ran);
    CHECK(calls == 1);
    CHECK(pointer_matched);
    CHECK(count_matched);
    return true;
}

bool test_asr_not_called_for_rejected_learned_file() {
    const std::vector<float> audio(16000, 0.1f);
    int calls = 0;
    const auto execution = pipeline::transcribe_admitted_file_once(
        false,
        true,
        audio.data(),
        audio.size(),
        [&](const float*, int) {
            ++calls;
            return pipeline::AsrResult{};
        });
    CHECK(!execution.ran);
    CHECK(calls == 0);
    return true;
}

bool test_quality_summary_log_fields_and_default_feature_privacy() {
    const std::string tsv_path = "/tmp/test_quality_summary.tsv";
    const std::string json_path = "/tmp/test_quality_summary.json";
    pipeline::Logger logger;
    CHECK(logger.open(tsv_path));
    logger.set_json_path(json_path);
    pipeline::RunConfig config;
    config.input_path = "test.wav";
    config.quality_policy = "learned";
    config.quality_threshold = 0.3;
    logger.log_run_start(config);

    pipeline::QualitySummaryRecord summary;
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
    logger.log_run_end(2, 1, 1, 0, 10.0, 100.0);
    logger.close();

    auto read_file = [](const std::string& path) {
        std::ifstream input(path);
        return std::string(
            std::istreambuf_iterator<char>(input),
            std::istreambuf_iterator<char>());
    };
    const std::string tsv = read_file(tsv_path);
    const std::string json = read_file(json_path);
    CHECK(tsv.find("event=quality_summary") != std::string::npos);
    CHECK(tsv.find("policy=learned") != std::string::npos);
    CHECK(tsv.find("chunk_count=2") != std::string::npos);
    CHECK(tsv.find("schema_version=quality-file-features-v1") != std::string::npos);
    CHECK(tsv.find("learned_raw_score=-0.25") != std::string::npos);
    CHECK(tsv.find("learned_probability=") != std::string::npos);
    CHECK(tsv.find("rule_summary=reject") != std::string::npos);
    CHECK(tsv.find("final_admission=admit") != std::string::npos);
    CHECK(tsv.find("asr_ran=1") != std::string::npos);
    CHECK(tsv.find("quality_features=") == std::string::npos);
    CHECK(json.find("\"quality\"") != std::string::npos);
    CHECK(json.find("\"learned_raw_score\"") != std::string::npos);
    CHECK(json.find("\"features\"") == std::string::npos);
    return true;
}

void benchmark_aggregation() {
    constexpr std::size_t iterations = 200000;
    const auto metrics = metrics_from(0.01);
    volatile float sink = 0.0f;
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t index = 0; index < iterations; ++index) {
        pipeline::QualityFeatureAggregator aggregator;
        aggregator.add(metrics, pipeline::GateDecision::PASS);
        aggregator.add(metrics, pipeline::GateDecision::BORDERLINE);
        aggregator.add(metrics, pipeline::GateDecision::FAIL);
        sink += aggregator.features()[0];
    }
    const auto end = std::chrono::steady_clock::now();
    const double total_ns =
        std::chrono::duration<double, std::nano>(end - start).count();
    std::cout << "aggregation_chunks=" << iterations * 3 << "\n";
    std::cout << "aggregation_state_bytes="
              << sizeof(pipeline::QualityFeatureAggregator) << "\n";
    std::cout << "aggregation_mean_ns_per_chunk="
              << total_ns / static_cast<double>(iterations * 3) << "\n";
    std::cout << "aggregation_benchmark_sink=" << sink << "\n";
}

}  // namespace

int main() {
    std::cout << "Running quality-gate runtime tests...\n\n";
    run("exact_feature_order_single_chunk", test_exact_feature_order_single_chunk);
    run("multi_chunk_means_maxima_fractions", test_multi_chunk_means_maxima_and_fractions);
    run("python_six_decimal_record_parity", test_python_six_decimal_record_parity);
    run("final_partial_chunk_aggregated", test_final_partial_chunk_is_aggregated);
    run("empty_and_reset", test_empty_and_reset);
    run("non_finite_metric_rejected", test_non_finite_metric_rejected);
    run("default_rule_behavior_unchanged", test_default_rule_behavior_unchanged);
    run("learned_threshold_boundary_alternate", test_learned_threshold_boundary_and_alternate);
    run("learned_accept_reject", test_learned_accept_and_reject);
    run("hybrid_truth_table", test_hybrid_truth_table);
    run("policy_threshold_validation", test_policy_and_threshold_validation);
    run("asr_called_once_accepted", test_asr_called_once_for_accepted_learned_file);
    run("asr_not_called_rejected", test_asr_not_called_for_rejected_learned_file);
    run("quality_summary_log_fields", test_quality_summary_log_fields_and_default_feature_privacy);
    benchmark_aggregation();
    std::cout << "\n" << passed << " passed, " << failed << " failed\n";
    return failed == 0 ? 0 : 1;
}
