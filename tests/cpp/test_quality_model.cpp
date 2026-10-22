// Tree traversal, ensemble accumulation, logistic link and input validation of the
// native quality model.

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>

#include <gtest/gtest.h>

#include "edge_audio/quality/quality_model.h"

namespace {

template <typename Function>
bool throws_invalid_argument(Function function) {
    try {
        function();
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

constexpr std::array<edge_audio::QualityTreeNode, 3> kBranchTree{{
    {1, 2, 0, 1.5, 0.0},
    {-1, -1, -2, -2.0, -2.0},
    {-1, -1, -2, -2.0, 3.0},
}};

TEST(QualityModel, LeftBranchTraversal) {
    const std::array<float, 1> features{{1.0f}};
    ASSERT_EQ(edge_audio::quality_model_detail::evaluate_tree(
                  kBranchTree.data(), kBranchTree.size(), features.data(), features.size()),
              -2.0);
}

TEST(QualityModel, RightBranchTraversal) {
    const std::array<float, 1> features{{2.0f}};
    ASSERT_EQ(edge_audio::quality_model_detail::evaluate_tree(
                  kBranchTree.data(), kBranchTree.size(), features.data(), features.size()),
              3.0);
}

TEST(QualityModel, BoundaryEqualityGoesLeft) {
    const std::array<float, 1> features{{1.5f}};
    ASSERT_EQ(edge_audio::quality_model_detail::evaluate_tree(
                  kBranchTree.data(), kBranchTree.size(), features.data(), features.size()),
              -2.0);
}

TEST(QualityModel, LeafEvaluation) {
    constexpr std::array<edge_audio::QualityTreeNode, 1> leaf{{
        {-1, -1, -2, -2.0, 4.25},
    }};
    const std::array<float, 1> features{{0.0f}};
    ASSERT_EQ(edge_audio::quality_model_detail::evaluate_tree(leaf.data(), leaf.size(),
                                                              features.data(), features.size()),
              4.25);
}

TEST(QualityModel, EnsembleAccumulation) {
    constexpr std::array<edge_audio::QualityTreeNode, 2> nodes{{
        {-1, -1, -2, -2.0, 2.0},
        {-1, -1, -2, -2.0, -1.0},
    }};
    constexpr std::array<edge_audio::QualityTreeRange, 2> trees{{
        {0, 1},
        {1, 1},
    }};
    const std::array<float, 1> features{{0.0f}};
    const double raw = edge_audio::quality_model_detail::accumulate_ensemble(
        0.5, 0.1, nodes.data(), nodes.size(), trees.data(), trees.size(), features.data(),
        features.size());
    ASSERT_TRUE(std::fabs(raw - 0.6) < 1e-15);
}

TEST(QualityModel, LogisticConversion) {
    ASSERT_EQ(edge_audio::quality_model_detail::logistic_probability(0.0), 0.5);
    const double probability =
        edge_audio::quality_model_detail::logistic_probability(std::log(3.0));
    ASSERT_TRUE(std::fabs(probability - 0.75) < 1e-15);
    ASSERT_EQ(edge_audio::quality_model_detail::logistic_probability(1000.0), 1.0);
    ASSERT_EQ(edge_audio::quality_model_detail::logistic_probability(-1000.0), 0.0);
}

TEST(QualityModel, DefaultAndAlternateThresholds) {
    std::array<float, edge_audio::kQualityFeatureCount> features{};
    const auto default_prediction = edge_audio::predict_quality(features);
    ASSERT_EQ(default_prediction.should_transcribe, (default_prediction.probability >= 0.3));
    ASSERT_TRUE(edge_audio::predict_quality(features, 0.0).should_transcribe);
    ASSERT_FALSE(edge_audio::predict_quality(features, 1.0).should_transcribe);
}

TEST(QualityModel, InvalidFeatureCount) {
    std::array<float, edge_audio::kQualityFeatureCount> features{};
    ASSERT_TRUE(throws_invalid_argument(
        [&] { edge_audio::predict_quality(features.data(), features.size() - 1); }));
    ASSERT_TRUE(
        throws_invalid_argument([&] { edge_audio::predict_quality(nullptr, features.size()); }));
}

TEST(QualityModel, NonFiniteFeaturesRejected) {
    std::array<float, edge_audio::kQualityFeatureCount> features{};
    features[0] = std::numeric_limits<float>::quiet_NaN();
    ASSERT_TRUE(throws_invalid_argument([&] { edge_audio::predict_quality(features); }));
    features[0] = std::numeric_limits<float>::infinity();
    ASSERT_TRUE(throws_invalid_argument([&] { edge_audio::predict_quality(features); }));
    features[0] = -std::numeric_limits<float>::infinity();
    ASSERT_TRUE(throws_invalid_argument([&] { edge_audio::predict_quality(features); }));
}

TEST(QualityModel, SchemaAndModelIdentity) {
    constexpr std::array<std::string_view, edge_audio::kQualityFeatureCount> expected_names{{
        "rms_mean",       "silence_ratio_mean",   "clipping_ratio_mean", "zcr_mean",
        "flatness_mean",  "centroid_hz_mean",     "rolloff_hz_mean",     "flux_mean",
        "band_low_mean",  "band_mid_mean",        "band_high_mean",      "active_frac_mean",
        "rms_max",        "silence_ratio_max",    "clipping_ratio_max",  "zcr_max",
        "flatness_max",   "centroid_hz_max",      "rolloff_hz_max",      "flux_max",
        "band_low_max",   "band_mid_max",         "band_high_max",       "active_frac_max",
        "gate_pass_frac", "gate_borderline_frac", "gate_fail_frac",
    }};
    ASSERT_EQ(edge_audio::quality_feature_count(), 27);
    ASSERT_EQ(edge_audio::quality_feature_names(), expected_names);
    ASSERT_EQ(edge_audio::quality_schema_version(), "quality-file-features-v1");
    ASSERT_EQ(edge_audio::quality_model_sha256(),
              "045481f2ca42739495e814573c2575cb5cfd8520d521b4a4d5ba2df4d4f4358b");
    ASSERT_EQ(edge_audio::quality_default_threshold(), 0.3);
    ASSERT_TRUE(edge_audio::quality_model_data_size_bytes() > 0);
}

}  // namespace
