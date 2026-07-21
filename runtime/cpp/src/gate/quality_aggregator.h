#pragma once

#include <array>
#include <cstddef>

#include "gate/gate.h"
#include "gate/quality_model.h"

namespace pipeline {

// Builds one quality-file-features-v1 vector from all non-overlapping
// five-second analysis chunks in a completed file.
class QualityFeatureAggregator {
public:
    void add(const GateMetrics& metrics, GateDecision decision);

    std::array<float, kQualityFeatureCount> features() const;
    std::size_t chunk_count() const noexcept { return chunk_count_; }
    bool empty() const noexcept { return chunk_count_ == 0; }
    void reset() noexcept;

private:
    static constexpr std::size_t kMetricCount = 12;

    std::array<double, kMetricCount> sums_{};
    std::array<double, kMetricCount> maxima_{};
    std::size_t chunk_count_ = 0;
    std::size_t pass_count_ = 0;
    std::size_t borderline_count_ = 0;
    std::size_t fail_count_ = 0;
};

}  // namespace pipeline
