// benchmark_quality: latency of the learned admission path, excluding ASR.
//
//   benchmark_quality [iterations]
//
// Reports per-call latency of feature aggregation (per 5 s chunk) and of the
// native gradient-boosted model, plus the model's static data size.

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <vector>

#include "edge_audio/quality/file_features.h"
#include "edge_audio/quality/quality_model.h"

namespace ea = edge_audio;
using Clock = std::chrono::steady_clock;

namespace {

struct Stats {
    double mean_ns;
    double p50_ns;
    double p95_ns;
};

Stats summarize(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    const double mean = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
    return {mean, samples[samples.size() / 2], samples[samples.size() * 95 / 100]};
}

ea::GateMetrics typical_speech_metrics() {
    ea::GateMetrics m;
    m.rms = 0.05;
    m.silence_ratio = 0.2;
    m.zcr = 900.0;
    m.spectral_flatness = 0.15;
    m.spectral_centroid = 1400.0;
    m.spectral_rolloff = 3200.0;
    m.spectral_flux = 2.5;
    m.band_energy_low = 0.3;
    m.band_energy_mid = 0.6;
    m.band_energy_high = 0.1;
    m.active_frame_frac = 0.9;
    return m;
}

}  // namespace

int main(int argc, char** argv) {
    const std::size_t iterations = argc > 1 ? std::strtoul(argv[1], nullptr, 10) : 100000;
    if (iterations == 0) {
        std::cerr << "usage: benchmark_quality [iterations > 0]\n";
        return 2;
    }

    const ea::GateMetrics metrics = typical_speech_metrics();
    ea::QualityFeatureAggregator aggregator;
    aggregator.add(metrics, ea::GateDecision::PASS);
    aggregator.add(metrics, ea::GateDecision::BORDERLINE);
    const auto features = aggregator.features();

    volatile double sink = 0.0;
    std::vector<double> aggregate_ns;
    std::vector<double> predict_ns;
    aggregate_ns.reserve(iterations);
    predict_ns.reserve(iterations);

    for (std::size_t i = 0; i < iterations / 10; ++i) {  // warm-up
        sink = sink + ea::predict_quality(features).raw_score;
    }
    for (std::size_t i = 0; i < iterations; ++i) {
        ea::QualityFeatureAggregator a;
        const auto t0 = Clock::now();
        a.add(metrics, ea::GateDecision::PASS);
        const auto t1 = Clock::now();
        sink = sink + a.features()[0];
        aggregate_ns.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count());

        const auto t2 = Clock::now();
        const auto prediction = ea::predict_quality(features);
        const auto t3 = Clock::now();
        sink = sink + prediction.raw_score;
        predict_ns.push_back(std::chrono::duration<double, std::nano>(t3 - t2).count());
    }

    const Stats agg = summarize(std::move(aggregate_ns));
    const Stats model = summarize(std::move(predict_ns));
    std::cout << "iterations=" << iterations << '\n'
              << "aggregate_chunk_mean_ns=" << agg.mean_ns << '\n'
              << "predict_mean_ns=" << model.mean_ns << '\n'
              << "predict_p50_ns=" << model.p50_ns << '\n'
              << "predict_p95_ns=" << model.p95_ns << '\n'
              << "predictions_per_second=" << 1e9 / model.mean_ns << '\n'
              << "model_data_bytes=" << ea::quality_model_data_size_bytes() << '\n';
    return sink == 12345.0 ? 1 : 0;  // keep `sink` observable
}
