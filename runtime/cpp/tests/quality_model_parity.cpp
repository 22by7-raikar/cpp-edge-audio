#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "gate/quality_model.h"

namespace {

constexpr std::array<char, 8> kMagic{{'Q', 'M', 'P', 'A', 'R', '1', '\0', '\0'}};
constexpr std::uint32_t kCorpusVersion = 1;
constexpr std::size_t kWarmupPredictions = 10000;
constexpr std::size_t kBenchmarkPredictions = 100000;

void read_exact(std::istream& input, char* output, std::size_t size) {
    input.read(output, static_cast<std::streamsize>(size));
    if (!input || static_cast<std::size_t>(input.gcount()) != size) {
        throw std::runtime_error("truncated quality parity corpus");
    }
}

std::uint8_t read_u8(std::istream& input) {
    char byte = 0;
    read_exact(input, &byte, 1);
    return static_cast<std::uint8_t>(static_cast<unsigned char>(byte));
}

std::uint32_t read_u32_le(std::istream& input) {
    std::array<unsigned char, 4> bytes{};
    read_exact(input, reinterpret_cast<char*>(bytes.data()), bytes.size());
    return static_cast<std::uint32_t>(bytes[0]) |
           (static_cast<std::uint32_t>(bytes[1]) << 8u) |
           (static_cast<std::uint32_t>(bytes[2]) << 16u) |
           (static_cast<std::uint32_t>(bytes[3]) << 24u);
}

std::uint64_t read_u64_le(std::istream& input) {
    std::array<unsigned char, 8> bytes{};
    read_exact(input, reinterpret_cast<char*>(bytes.data()), bytes.size());
    std::uint64_t value = 0;
    for (std::size_t index = 0; index < bytes.size(); ++index) {
        value |= static_cast<std::uint64_t>(bytes[index]) << (8u * index);
    }
    return value;
}

float read_float_le(std::istream& input) {
    const std::uint32_t bits = read_u32_le(input);
    float value = 0.0f;
    static_assert(sizeof(value) == sizeof(bits), "float must be IEEE binary32");
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

double read_double_le(std::istream& input) {
    const std::uint64_t bits = read_u64_le(input);
    double value = 0.0;
    static_assert(sizeof(value) == sizeof(bits), "double must be IEEE binary64");
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

std::string fixed_string(const std::vector<char>& bytes) {
    const auto end = std::find(bytes.begin(), bytes.end(), '\0');
    return std::string(bytes.begin(), end);
}

double percentile(const std::vector<double>& sorted, double fraction) {
    const std::size_t index = static_cast<std::size_t>(
        fraction * static_cast<double>(sorted.size() - 1));
    return sorted[index];
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: quality_model_parity <corpus.bin>\n";
        return 2;
    }

    try {
        std::ifstream input(argv[1], std::ios::binary);
        if (!input) {
            throw std::runtime_error("cannot open quality parity corpus");
        }

        std::array<char, 8> magic{};
        read_exact(input, magic.data(), magic.size());
        const std::uint32_t version = read_u32_le(input);
        const std::uint32_t feature_count = read_u32_le(input);
        const std::uint32_t validation_count = read_u32_le(input);
        const std::uint32_t test_count = read_u32_le(input);
        std::vector<char> schema_bytes(32);
        std::vector<char> hash_bytes(64);
        read_exact(input, schema_bytes.data(), schema_bytes.size());
        read_exact(input, hash_bytes.data(), hash_bytes.size());

        if (magic != kMagic || version != kCorpusVersion) {
            throw std::runtime_error("quality parity corpus header mismatch");
        }
        if (feature_count != pipeline::quality_feature_count()) {
            throw std::runtime_error("quality parity feature count mismatch");
        }
        if (fixed_string(schema_bytes) != pipeline::quality_schema_version()) {
            throw std::runtime_error("quality parity schema mismatch");
        }
        if (std::string(hash_bytes.begin(), hash_bytes.end()) !=
            pipeline::quality_model_sha256()) {
            throw std::runtime_error("quality parity model hash mismatch");
        }

        const std::size_t example_count =
            static_cast<std::size_t>(validation_count) + test_count;
        double raw_sum = 0.0;
        double probability_sum = 0.0;
        double max_raw_difference = 0.0;
        double max_probability_difference = 0.0;
        std::string worst_raw_id;
        std::string worst_probability_id;
        std::size_t disagreements = 0;
        std::array<float, pipeline::kQualityFeatureCount> benchmark_features{};

        for (std::size_t row = 0; row < example_count; ++row) {
            std::array<char, 24> id_bytes{};
            read_exact(input, id_bytes.data(), id_bytes.size());
            const std::string example_id(id_bytes.begin(), id_bytes.end());
            std::array<float, pipeline::kQualityFeatureCount> features{};
            for (float& value : features) {
                value = read_float_le(input);
            }
            const double expected_raw = read_double_le(input);
            const double expected_probability = read_double_le(input);
            const bool expected_decision = read_u8(input) != 0;
            const pipeline::QualityPrediction prediction =
                pipeline::predict_quality(features, 0.3);

            const double raw_difference =
                std::fabs(prediction.raw_score - expected_raw);
            const double probability_difference =
                std::fabs(prediction.probability - expected_probability);
            raw_sum += raw_difference;
            probability_sum += probability_difference;
            if (row == 0 || raw_difference > max_raw_difference) {
                max_raw_difference = raw_difference;
                worst_raw_id = example_id;
            }
            if (row == 0 || probability_difference > max_probability_difference) {
                max_probability_difference = probability_difference;
                worst_probability_id = example_id;
            }
            if (prediction.should_transcribe != expected_decision) {
                ++disagreements;
            }
            if (row == 0) {
                benchmark_features = features;
            }
        }
        if (input.peek() != std::char_traits<char>::eof()) {
            throw std::runtime_error("quality parity corpus has trailing bytes");
        }

        volatile double benchmark_sink = 0.0;
        for (std::size_t index = 0; index < kWarmupPredictions; ++index) {
            benchmark_sink += pipeline::predict_quality(benchmark_features).raw_score;
        }
        std::vector<double> latencies_ns;
        latencies_ns.reserve(kBenchmarkPredictions);
        for (std::size_t index = 0; index < kBenchmarkPredictions; ++index) {
            const auto start = std::chrono::steady_clock::now();
            const auto prediction = pipeline::predict_quality(benchmark_features);
            const auto end = std::chrono::steady_clock::now();
            benchmark_sink += prediction.raw_score;
            latencies_ns.push_back(
                std::chrono::duration<double, std::nano>(end - start).count());
        }
        std::sort(latencies_ns.begin(), latencies_ns.end());
        const double mean_latency_ns = std::accumulate(
            latencies_ns.begin(), latencies_ns.end(), 0.0) /
            static_cast<double>(latencies_ns.size());

        std::cout << std::setprecision(17);
        std::cout << "examples=" << example_count << "\n";
        std::cout << "validation_examples=" << validation_count << "\n";
        std::cout << "test_examples=" << test_count << "\n";
        std::cout << "max_abs_raw_score_difference=" << max_raw_difference << "\n";
        std::cout << "mean_abs_raw_score_difference="
                  << raw_sum / static_cast<double>(example_count) << "\n";
        std::cout << "max_abs_probability_difference="
                  << max_probability_difference << "\n";
        std::cout << "mean_abs_probability_difference="
                  << probability_sum / static_cast<double>(example_count) << "\n";
        std::cout << "decision_agreement="
                  << (example_count - disagreements) << "/" << example_count << "\n";
        std::cout << "disagreements=" << disagreements << "\n";
        std::cout << "worst_raw_score_example_id=" << worst_raw_id << "\n";
        std::cout << "worst_probability_example_id=" << worst_probability_id << "\n";
        std::cout << "benchmark_predictions=" << kBenchmarkPredictions << "\n";
        std::cout << "mean_latency_ns=" << mean_latency_ns << "\n";
        std::cout << "p50_latency_ns=" << percentile(latencies_ns, 0.50) << "\n";
        std::cout << "p95_latency_ns=" << percentile(latencies_ns, 0.95) << "\n";
        std::cout << "predictions_per_second=" << 1.0e9 / mean_latency_ns << "\n";
        std::cout << "model_data_size_bytes="
                  << pipeline::quality_model_data_size_bytes() << "\n";
        std::cout << "benchmark_sink=" << benchmark_sink << "\n";

        const bool parity_passed =
            disagreements == 0 && max_raw_difference <= 1e-12 &&
            max_probability_difference <= 1e-10;
        return parity_passed ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "quality parity error: " << error.what() << "\n";
        return 2;
    }
}
