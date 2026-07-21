#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace pipeline {

inline constexpr std::size_t kQualityFeatureCount = 27;
inline constexpr double kQualityDefaultThreshold = 0.3;

struct QualityPrediction {
    double raw_score = 0.0;
    double probability = 0.0;
    bool should_transcribe = false;
};

struct QualityTreeNode {
    std::int16_t children_left;
    std::int16_t children_right;
    std::int8_t feature;
    double threshold;
    double value;
};

struct QualityTreeRange {
    std::uint16_t node_offset;
    std::uint8_t node_count;
};

namespace quality_model_detail {

double evaluate_tree(
    const QualityTreeNode* nodes,
    std::size_t node_count,
    const float* features,
    std::size_t feature_count);

double accumulate_ensemble(
    double initial_raw_score,
    double learning_rate,
    const QualityTreeNode* nodes,
    std::size_t total_node_count,
    const QualityTreeRange* trees,
    std::size_t tree_count,
    const float* features,
    std::size_t feature_count);

double logistic_probability(double raw_score) noexcept;

}  // namespace quality_model_detail

QualityPrediction predict_quality(
    const std::array<float, kQualityFeatureCount>& features,
    double decision_threshold = kQualityDefaultThreshold);

QualityPrediction predict_quality(
    const float* features,
    std::size_t feature_count,
    double decision_threshold = kQualityDefaultThreshold);

std::size_t quality_feature_count() noexcept;
double quality_default_threshold() noexcept;
std::string_view quality_schema_version() noexcept;
std::string_view quality_model_sha256() noexcept;
const std::array<std::string_view, kQualityFeatureCount>&
quality_feature_names() noexcept;
std::size_t quality_model_data_size_bytes() noexcept;

}  // namespace pipeline
