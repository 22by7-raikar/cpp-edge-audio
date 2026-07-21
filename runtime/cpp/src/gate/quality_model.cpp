#include "gate/quality_model.h"

#include <cmath>
#include <stdexcept>

#include "quality_model_neura_v1.h"

namespace pipeline {
namespace quality_model_detail {

double evaluate_tree(
    const QualityTreeNode* nodes,
    std::size_t node_count,
    const float* features,
    std::size_t feature_count) {
    if (nodes == nullptr || node_count == 0) {
        throw std::invalid_argument("quality tree is empty");
    }
    if (features == nullptr || feature_count == 0) {
        throw std::invalid_argument("quality feature vector is empty");
    }

    std::size_t node_index = 0;
    for (std::size_t visited = 0; visited < node_count; ++visited) {
        const QualityTreeNode& node = nodes[node_index];
        const bool is_leaf = node.children_left < 0;
        if (is_leaf) {
            if (node.children_right >= 0) {
                throw std::logic_error("malformed quality tree leaf");
            }
            return node.value;
        }

        if (node.children_right < 0 || node.feature < 0 ||
            static_cast<std::size_t>(node.feature) >= feature_count) {
            throw std::logic_error("malformed quality tree split");
        }
        const std::int16_t next =
            features[static_cast<std::size_t>(node.feature)] <= node.threshold
                ? node.children_left
                : node.children_right;
        if (next < 0 || static_cast<std::size_t>(next) >= node_count) {
            throw std::logic_error("quality tree child index is out of range");
        }
        node_index = static_cast<std::size_t>(next);
    }
    throw std::logic_error("quality tree traversal did not reach a leaf");
}

double accumulate_ensemble(
    double initial_raw_score,
    double learning_rate,
    const QualityTreeNode* nodes,
    std::size_t total_node_count,
    const QualityTreeRange* trees,
    std::size_t tree_count,
    const float* features,
    std::size_t feature_count) {
    if (nodes == nullptr || trees == nullptr) {
        throw std::invalid_argument("quality ensemble data is missing");
    }

    double raw_score = initial_raw_score;
    for (std::size_t index = 0; index < tree_count; ++index) {
        const QualityTreeRange& tree = trees[index];
        const std::size_t offset = tree.node_offset;
        const std::size_t count = tree.node_count;
        if (count == 0 || offset > total_node_count ||
            count > total_node_count - offset) {
            throw std::logic_error("quality tree range is out of bounds");
        }
        raw_score += learning_rate * evaluate_tree(
            nodes + offset, count, features, feature_count);
    }
    return raw_score;
}

double logistic_probability(double raw_score) noexcept {
    if (raw_score >= 0.0) {
        const double exp_negative = std::exp(-raw_score);
        return 1.0 / (1.0 + exp_negative);
    }
    const double exp_positive = std::exp(raw_score);
    return exp_positive / (1.0 + exp_positive);
}

}  // namespace quality_model_detail

QualityPrediction predict_quality(
    const std::array<float, kQualityFeatureCount>& features,
    double decision_threshold) {
    return predict_quality(features.data(), features.size(), decision_threshold);
}

QualityPrediction predict_quality(
    const float* features,
    std::size_t feature_count,
    double decision_threshold) {
    if (features == nullptr) {
        throw std::invalid_argument("quality features must not be null");
    }
    if (feature_count != kQualityFeatureCount) {
        throw std::invalid_argument("quality model requires exactly 27 features");
    }
    if (!std::isfinite(decision_threshold) || decision_threshold < 0.0 ||
        decision_threshold > 1.0) {
        throw std::invalid_argument("quality decision threshold must be in [0, 1]");
    }
    for (std::size_t index = 0; index < feature_count; ++index) {
        if (!std::isfinite(static_cast<double>(features[index]))) {
            throw std::invalid_argument("quality features must all be finite");
        }
    }

    static_assert(
        quality_model_generated::kFeatureNames.size() == kQualityFeatureCount,
        "generated quality feature count differs from native API");
    static_assert(
        quality_model_generated::kSelectedThreshold == kQualityDefaultThreshold,
        "generated quality threshold differs from native API");

    const double raw_score = quality_model_detail::accumulate_ensemble(
        quality_model_generated::kInitialRawScore,
        quality_model_generated::kLearningRate,
        quality_model_generated::kNodes.data(),
        quality_model_generated::kNodes.size(),
        quality_model_generated::kTrees.data(),
        quality_model_generated::kTrees.size(),
        features,
        feature_count);
    const double probability =
        quality_model_detail::logistic_probability(raw_score);
    return {raw_score, probability, probability >= decision_threshold};
}

std::size_t quality_feature_count() noexcept {
    return kQualityFeatureCount;
}

double quality_default_threshold() noexcept {
    return quality_model_generated::kSelectedThreshold;
}

std::string_view quality_schema_version() noexcept {
    return quality_model_generated::kSchemaVersion;
}

std::string_view quality_model_sha256() noexcept {
    return quality_model_generated::kModelSha256;
}

const std::array<std::string_view, kQualityFeatureCount>&
quality_feature_names() noexcept {
    return quality_model_generated::kFeatureNames;
}

std::size_t quality_model_data_size_bytes() noexcept {
    return sizeof(quality_model_generated::kNodes) +
           sizeof(quality_model_generated::kTrees);
}

}  // namespace pipeline
