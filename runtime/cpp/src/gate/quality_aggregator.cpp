#include "gate/quality_aggregator.h"

#include <cmath>
#include <stdexcept>

namespace pipeline {
namespace {

constexpr std::size_t kMetricCount = 12;

std::array<double, kMetricCount> ordered_metrics(const GateMetrics& metrics) {
    return {{
        metrics.rms,
        metrics.silence_ratio,
        metrics.clipping_ratio,
        metrics.zcr,
        metrics.spectral_flatness,
        metrics.spectral_centroid,
        metrics.spectral_rolloff,
        metrics.spectral_flux,
        metrics.band_energy_low,
        metrics.band_energy_mid,
        metrics.band_energy_high,
        metrics.active_frame_frac,
    }};
}

// The authoritative Python extractor stores each chunk metric rounded to six
// decimal places before file-level mean/max aggregation.
double python_record_value(double value) {
    if (!std::isfinite(value)) {
        throw std::invalid_argument("quality chunk metrics must be finite");
    }
    return std::round(value * 1000000.0) / 1000000.0;
}

}  // namespace

void QualityFeatureAggregator::add(
    const GateMetrics& metrics,
    GateDecision decision) {
    auto values = ordered_metrics(metrics);
    for (double& value : values) {
        value = python_record_value(value);
    }
    for (std::size_t index = 0; index < values.size(); ++index) {
        const double value = values[index];
        sums_[index] += value;
        if (chunk_count_ == 0 || value > maxima_[index]) {
            maxima_[index] = value;
        }
    }

    switch (decision) {
        case GateDecision::PASS:       ++pass_count_; break;
        case GateDecision::BORDERLINE: ++borderline_count_; break;
        case GateDecision::FAIL:       ++fail_count_; break;
    }
    ++chunk_count_;
}

std::array<float, kQualityFeatureCount>
QualityFeatureAggregator::features() const {
    if (empty()) {
        throw std::logic_error("cannot aggregate an empty quality file");
    }

    std::array<float, kQualityFeatureCount> output{};
    const double count = static_cast<double>(chunk_count_);
    for (std::size_t index = 0; index < sums_.size(); ++index) {
        output[index] = static_cast<float>(sums_[index] / count);
        output[index + sums_.size()] = static_cast<float>(maxima_[index]);
    }
    output[24] = static_cast<float>(static_cast<double>(pass_count_) / count);
    output[25] = static_cast<float>(
        static_cast<double>(borderline_count_) / count);
    output[26] = static_cast<float>(static_cast<double>(fail_count_) / count);
    return output;
}

void QualityFeatureAggregator::reset() noexcept {
    sums_.fill(0.0);
    maxima_.fill(0.0);
    chunk_count_ = 0;
    pass_count_ = 0;
    borderline_count_ = 0;
    fail_count_ = 0;
}

}  // namespace pipeline
