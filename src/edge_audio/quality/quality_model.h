/// @file
/// @brief Native inference for the exported gradient-boosted quality model.
/// @ingroup quality

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace edge_audio {

/// @addtogroup quality
/// @{

/// @brief Length of the `quality-file-features-v1` vector.
inline constexpr std::size_t kQualityFeatureCount = 27;

/// @brief Decision threshold selected on the validation split (never on test).
inline constexpr double kQualityDefaultThreshold = 0.3;

/// @brief Model output for one feature vector.
struct QualityPrediction {
    double raw_score = 0.0;          ///< Ensemble log-odds.
    double probability = 0.0;        ///< Logistic of #raw_score: P(worth transcribing).
    bool should_transcribe = false;  ///< `probability >= threshold`.
};

/// @brief One node of an exported regression tree (leaf when `children_left < 0`).
struct QualityTreeNode {
    std::int16_t children_left;   ///< Index of the `<=` child within the tree, or -1.
    std::int16_t children_right;  ///< Index of the `>` child within the tree, or -1.
    std::int8_t feature;          ///< Feature index tested at this split.
    double threshold;             ///< Split threshold (`feature <= threshold` goes left).
    double value;                 ///< Leaf value.
};

/// @brief Location of one tree in the flat node array.
struct QualityTreeRange {
    std::uint16_t node_offset;  ///< First node of the tree.
    std::uint8_t node_count;    ///< Nodes in the tree.
};

/// @brief Building blocks of predict_quality(), exposed for unit tests.
namespace quality_model_detail {

/// @brief Walk one tree to a leaf.
/// @throws std::logic_error for a malformed tree, std::invalid_argument for empty inputs.
double evaluate_tree(const QualityTreeNode* nodes, std::size_t node_count, const float* features,
                     std::size_t feature_count);

/// @brief `initial + learning_rate * sum(tree outputs)`, as in scikit-learn.
double accumulate_ensemble(double initial_raw_score, double learning_rate,
                           const QualityTreeNode* nodes, std::size_t total_node_count,
                           const QualityTreeRange* trees, std::size_t tree_count,
                           const float* features, std::size_t feature_count);

/// @brief Numerically stable logistic function.
double logistic_probability(double raw_score) noexcept;

}  // namespace quality_model_detail

/// @brief Score a feature vector with the compiled-in model.
///
/// The model (100 depth-3 trees, ~36 KB of constant data) is generated from the
/// trained scikit-learn artifact by `tools/python/export/export_quality_model.py`
/// into `quality_model_data.h`. Inference allocates nothing and runs in about a
/// microsecond.
///
/// @param features           The 27 ordered features (see QualityFeatureAggregator).
/// @param decision_threshold Probability threshold in [0, 1].
/// @throws std::invalid_argument for a non-finite feature or an invalid threshold.
QualityPrediction predict_quality(const std::array<float, kQualityFeatureCount>& features,
                                  double decision_threshold = kQualityDefaultThreshold);

/// @overload
/// @throws std::invalid_argument if @p feature_count != kQualityFeatureCount.
QualityPrediction predict_quality(const float* features, std::size_t feature_count,
                                  double decision_threshold = kQualityDefaultThreshold);

/// @name Model identity
/// Metadata compiled in from the exported artifact.
/// @{
std::size_t quality_feature_count() noexcept;        ///< Always kQualityFeatureCount.
double quality_default_threshold() noexcept;         ///< Validation-selected threshold.
std::string_view quality_schema_version() noexcept;  ///< `quality-file-features-v1`.
std::string_view quality_model_sha256() noexcept;    ///< SHA-256 of the source artifact.
const std::array<std::string_view, kQualityFeatureCount>&
quality_feature_names() noexcept;                      ///< Ordered names.
std::size_t quality_model_data_size_bytes() noexcept;  ///< Size of node and tree tables.
/// @}

/// @}

}  // namespace edge_audio
