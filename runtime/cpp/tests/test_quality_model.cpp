#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>

#include "gate/quality_model.h"

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

template <typename Function>
bool throws_invalid_argument(Function function) {
    try {
        function();
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

constexpr std::array<pipeline::QualityTreeNode, 3> kBranchTree{{
    {1, 2, 0, 1.5, 0.0},
    {-1, -1, -2, -2.0, -2.0},
    {-1, -1, -2, -2.0, 3.0},
}};

bool test_left_branch_traversal() {
    const std::array<float, 1> features{{1.0f}};
    CHECK(pipeline::quality_model_detail::evaluate_tree(
              kBranchTree.data(), kBranchTree.size(),
              features.data(), features.size()) == -2.0);
    return true;
}

bool test_right_branch_traversal() {
    const std::array<float, 1> features{{2.0f}};
    CHECK(pipeline::quality_model_detail::evaluate_tree(
              kBranchTree.data(), kBranchTree.size(),
              features.data(), features.size()) == 3.0);
    return true;
}

bool test_boundary_equality_goes_left() {
    const std::array<float, 1> features{{1.5f}};
    CHECK(pipeline::quality_model_detail::evaluate_tree(
              kBranchTree.data(), kBranchTree.size(),
              features.data(), features.size()) == -2.0);
    return true;
}

bool test_leaf_evaluation() {
    constexpr std::array<pipeline::QualityTreeNode, 1> leaf{{
        {-1, -1, -2, -2.0, 4.25},
    }};
    const std::array<float, 1> features{{0.0f}};
    CHECK(pipeline::quality_model_detail::evaluate_tree(
              leaf.data(), leaf.size(), features.data(), features.size()) == 4.25);
    return true;
}

bool test_ensemble_accumulation() {
    constexpr std::array<pipeline::QualityTreeNode, 2> nodes{{
        {-1, -1, -2, -2.0, 2.0},
        {-1, -1, -2, -2.0, -1.0},
    }};
    constexpr std::array<pipeline::QualityTreeRange, 2> trees{{
        {0, 1},
        {1, 1},
    }};
    const std::array<float, 1> features{{0.0f}};
    const double raw = pipeline::quality_model_detail::accumulate_ensemble(
        0.5, 0.1, nodes.data(), nodes.size(), trees.data(), trees.size(),
        features.data(), features.size());
    CHECK(std::fabs(raw - 0.6) < 1e-15);
    return true;
}

bool test_logistic_conversion() {
    CHECK(pipeline::quality_model_detail::logistic_probability(0.0) == 0.5);
    const double probability = pipeline::quality_model_detail::logistic_probability(
        std::log(3.0));
    CHECK(std::fabs(probability - 0.75) < 1e-15);
    CHECK(pipeline::quality_model_detail::logistic_probability(1000.0) == 1.0);
    CHECK(pipeline::quality_model_detail::logistic_probability(-1000.0) == 0.0);
    return true;
}

bool test_default_and_alternate_thresholds() {
    std::array<float, pipeline::kQualityFeatureCount> features{};
    const auto default_prediction = pipeline::predict_quality(features);
    CHECK(default_prediction.should_transcribe ==
          (default_prediction.probability >= 0.3));
    CHECK(pipeline::predict_quality(features, 0.0).should_transcribe);
    CHECK(!pipeline::predict_quality(features, 1.0).should_transcribe);
    return true;
}

bool test_invalid_feature_count() {
    std::array<float, pipeline::kQualityFeatureCount> features{};
    CHECK(throws_invalid_argument([&] {
        pipeline::predict_quality(features.data(), features.size() - 1);
    }));
    CHECK(throws_invalid_argument([&] {
        pipeline::predict_quality(nullptr, features.size());
    }));
    return true;
}

bool test_non_finite_features_rejected() {
    std::array<float, pipeline::kQualityFeatureCount> features{};
    features[0] = std::numeric_limits<float>::quiet_NaN();
    CHECK(throws_invalid_argument([&] { pipeline::predict_quality(features); }));
    features[0] = std::numeric_limits<float>::infinity();
    CHECK(throws_invalid_argument([&] { pipeline::predict_quality(features); }));
    features[0] = -std::numeric_limits<float>::infinity();
    CHECK(throws_invalid_argument([&] { pipeline::predict_quality(features); }));
    return true;
}

bool test_schema_and_model_identity() {
    constexpr std::array<std::string_view, pipeline::kQualityFeatureCount>
        expected_names{{
            "rms_mean", "silence_ratio_mean", "clipping_ratio_mean", "zcr_mean",
            "flatness_mean", "centroid_hz_mean", "rolloff_hz_mean", "flux_mean",
            "band_low_mean", "band_mid_mean", "band_high_mean", "active_frac_mean",
            "rms_max", "silence_ratio_max", "clipping_ratio_max", "zcr_max",
            "flatness_max", "centroid_hz_max", "rolloff_hz_max", "flux_max",
            "band_low_max", "band_mid_max", "band_high_max", "active_frac_max",
            "gate_pass_frac", "gate_borderline_frac", "gate_fail_frac",
        }};
    CHECK(pipeline::quality_feature_count() == 27);
    CHECK(pipeline::quality_feature_names() == expected_names);
    CHECK(pipeline::quality_schema_version() == "quality-file-features-v1");
    CHECK(pipeline::quality_model_sha256() ==
          "045481f2ca42739495e814573c2575cb5cfd8520d521b4a4d5ba2df4d4f4358b");
    CHECK(pipeline::quality_default_threshold() == 0.3);
    CHECK(pipeline::quality_model_data_size_bytes() > 0);
    return true;
}

}  // namespace

int main() {
    std::cout << "Running native quality-model tests...\n\n";
    run("left_branch_traversal", test_left_branch_traversal);
    run("right_branch_traversal", test_right_branch_traversal);
    run("boundary_equality_goes_left", test_boundary_equality_goes_left);
    run("leaf_evaluation", test_leaf_evaluation);
    run("ensemble_accumulation", test_ensemble_accumulation);
    run("logistic_conversion", test_logistic_conversion);
    run("default_and_alternate_thresholds", test_default_and_alternate_thresholds);
    run("invalid_feature_count", test_invalid_feature_count);
    run("non_finite_features_rejected", test_non_finite_features_rejected);
    run("schema_and_model_identity", test_schema_and_model_identity);
    std::cout << "\n" << passed << " passed, " << failed << " failed\n";
    return failed == 0 ? 0 : 1;
}
