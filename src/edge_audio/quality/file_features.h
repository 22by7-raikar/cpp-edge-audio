/// @file
/// @brief Aggregation of chunk metrics into the learned model's 27-feature vector.
/// @ingroup quality

#pragma once

#include <array>
#include <cstddef>

#include "edge_audio/quality/quality_model.h"
#include "edge_audio/quality/rule_gate.h"

namespace edge_audio {

/// @addtogroup quality
/// @{

/// @brief Builds one `quality-file-features-v1` vector from the chunks of a
/// complete file or utterance.
///
/// Layout (see quality_feature_names()):
/// - [0, 12):  mean of each GateMetrics value except `duration_sec`;
/// - [12, 24): maximum of the same 12 values;
/// - [24, 27): fraction of chunks the rule gate marked PASS / BORDERLINE / FAIL.
///
/// Each metric is rounded to 6 decimal places before aggregation, matching the
/// Python feature extractor the model was trained on; this is what makes the
/// native and Python vectors bit-identical.
///
/// @pre Chunks are the non-overlapping 5 s windows used at training time.
class QualityFeatureAggregator {
public:
    /// @brief Accumulate one chunk.
    /// @throws std::invalid_argument if any metric is non-finite.
    void add(const GateMetrics& metrics, GateDecision decision);

    /// @brief The ordered feature vector.
    /// @throws std::logic_error if no chunk was added.
    std::array<float, kQualityFeatureCount> features() const;

    std::size_t chunk_count() const noexcept { return chunk_count_; }  ///< Chunks added.
    bool empty() const noexcept { return chunk_count_ == 0; }          ///< No chunk added yet.
    void reset() noexcept;  ///< Discard all accumulated chunks.

private:
    static constexpr std::size_t kMetricCount = 12;

    std::array<double, kMetricCount> sums_{};
    std::array<double, kMetricCount> maxima_{};
    std::size_t chunk_count_ = 0;
    std::size_t pass_count_ = 0;
    std::size_t borderline_count_ = 0;
    std::size_t fail_count_ = 0;
};

/// @}

}  // namespace edge_audio
