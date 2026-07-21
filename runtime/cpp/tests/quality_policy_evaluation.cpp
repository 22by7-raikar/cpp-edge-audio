#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "audio/audio_io.h"
#include "chunker/chunker.h"
#include "gate/gate.h"
#include "gate/quality_aggregator.h"
#include "gate/quality_model.h"
#include "scene/adaptive.h"
#include "scene/scene.h"

namespace {

struct Metrics {
    std::size_t tp = 0;
    std::size_t fp = 0;
    std::size_t tn = 0;
    std::size_t fn = 0;
    std::size_t asr_calls = 0;
};

std::string json_string(const std::string& line, const std::string& key) {
    const std::string prefix = "\"" + key + "\": \"";
    const std::size_t begin = line.find(prefix);
    if (begin == std::string::npos) {
        throw std::runtime_error("missing JSON string field: " + key);
    }
    const std::size_t value_begin = begin + prefix.size();
    const std::size_t end = line.find('"', value_begin);
    if (end == std::string::npos) {
        throw std::runtime_error("unterminated JSON string field: " + key);
    }
    return line.substr(value_begin, end - value_begin);
}

bool has_suffix(const std::string& value, const std::string& suffix) {
    return value.size() >= suffix.size() &&
           value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::vector<std::string> split_csv(const std::string& line) {
    std::vector<std::string> fields;
    std::size_t begin = 0;
    while (begin <= line.size()) {
        const std::size_t end = line.find(',', begin);
        fields.push_back(line.substr(
            begin, end == std::string::npos ? end : end - begin));
        if (end == std::string::npos) break;
        begin = end + 1;
    }
    return fields;
}

void observe(Metrics& metrics, bool target, bool prediction) {
    if (target && prediction) ++metrics.tp;
    else if (!target && prediction) ++metrics.fp;
    else if (!target && !prediction) ++metrics.tn;
    else ++metrics.fn;
}

void print_metrics(
    const char* policy,
    const Metrics& metrics,
    std::size_t ungated_calls) {
    const double far = static_cast<double>(metrics.fp) /
        static_cast<double>(metrics.fp + metrics.tn);
    const double frr = static_cast<double>(metrics.fn) /
        static_cast<double>(metrics.tp + metrics.fn);
    const double precision = static_cast<double>(metrics.tp) /
        static_cast<double>(metrics.tp + metrics.fp);
    const double recall = static_cast<double>(metrics.tp) /
        static_cast<double>(metrics.tp + metrics.fn);
    const double f1 = 2.0 * precision * recall / (precision + recall);
    const double avoided = static_cast<double>(ungated_calls - metrics.asr_calls) /
        static_cast<double>(ungated_calls);

    std::cout << policy
              << " tp=" << metrics.tp
              << " fp=" << metrics.fp
              << " tn=" << metrics.tn
              << " fn=" << metrics.fn
              << " far=" << far
              << " frr=" << frr
              << " f1=" << f1
              << " asr_invocations=" << metrics.asr_calls
              << " asr_calls_avoided_pct=" << avoided * 100.0
              << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 4) {
        std::cerr << "usage: quality_policy_evaluation <labels.jsonl> [repo-root] [saved-predictions.csv]\n";
        return 2;
    }
    const std::string labels_path = argv[1];
    const std::string repo_root = argc >= 3 ? argv[2] : ".";

    try {
        std::ifstream labels(labels_path);
        if (!labels) throw std::runtime_error("cannot open labels file");

        std::unordered_map<std::string, bool> expected_learned;
        if (argc == 4) {
            std::ifstream predictions(argv[3]);
            if (!predictions) {
                throw std::runtime_error("cannot open saved predictions file");
            }
            std::string csv_line;
            std::getline(predictions, csv_line);
            while (std::getline(predictions, csv_line)) {
                const auto fields = split_csv(csv_line);
                if (fields.size() != 9) {
                    throw std::runtime_error("unexpected saved-prediction CSV row");
                }
                expected_learned.emplace(fields[1], fields[7] == "1");
            }
        }

        Metrics rule;
        Metrics learned;
        Metrics hybrid;
        std::size_t examples = 0;
        std::size_t skipped_non_wav = 0;
        std::size_t chunk_count = 0;
        double audio_seconds = 0.0;
        double learned_inference_us = 0.0;
        std::size_t learned_agreements = 0;
        std::size_t learned_disagreements = 0;
        const auto wall_start = std::chrono::steady_clock::now();

        std::string line;
        while (std::getline(labels, line)) {
            if (line.empty()) continue;
            const std::string relative_path = json_string(line, "path");
            if (!has_suffix(relative_path, ".wav")) {
                ++skipped_non_wav;
                continue;
            }
            const bool target = json_string(line, "should_transcribe") == "yes";

            pipeline::AudioBuffer audio;
            std::string error;
            const std::string path = relative_path.front() == '/'
                ? relative_path
                : repo_root + "/" + relative_path;
            if (!pipeline::load_wav(path, audio, error)) {
                throw std::runtime_error(path + ": " + error);
            }
            if (audio.sample_rate != 16000) {
                audio = pipeline::resample(audio, 16000);
            }

            pipeline::ChunkerConfig chunk_config;
            chunk_config.chunk_ms = 5000;
            chunk_config.hop_ms = 0;
            const auto chunks = pipeline::chunk_audio(audio, chunk_config);
            if (chunks.empty()) {
                throw std::runtime_error(path + ": no analysis chunks");
            }

            pipeline::GateConfig gate_config;
            pipeline::SceneConfig scene_config;
            pipeline::AdaptiveConfig adaptive_config;
            pipeline::AdaptiveController adaptive(adaptive_config);
            pipeline::QualityFeatureAggregator aggregator;
            bool rule_file_admission = false;
            std::size_t rule_file_calls = 0;

            for (const auto& chunk : chunks) {
                const pipeline::GateResult base_gate =
                    pipeline::evaluate_chunk(chunk, gate_config);
                aggregator.add(base_gate.metrics, base_gate.decision);

                pipeline::GateResult rule_gate = base_gate;
                if (adaptive.dominant_scene() == pipeline::SceneLabel::NOISE) {
                    rule_gate = pipeline::evaluate_chunk(
                        chunk, adaptive.adapt_gate(gate_config));
                }
                const pipeline::SceneResult scene =
                    pipeline::classify(rule_gate.metrics, scene_config);
                const bool rule_chunk_admission =
                    (rule_gate.decision == pipeline::GateDecision::PASS ||
                     rule_gate.decision == pipeline::GateDecision::BORDERLINE) &&
                    !adaptive.skip_asr(scene.label);
                if (rule_chunk_admission) {
                    rule_file_admission = true;
                    ++rule_file_calls;
                }
                adaptive.push_scene(scene.label);
            }

            const auto features = aggregator.features();
            const auto inference_start = std::chrono::steady_clock::now();
            const pipeline::QualityPrediction quality =
                pipeline::predict_quality(features, 0.3);
            const auto inference_end = std::chrono::steady_clock::now();
            learned_inference_us +=
                std::chrono::duration<double, std::micro>(
                    inference_end - inference_start).count();
            const bool hybrid_admission =
                rule_file_admission && quality.should_transcribe;
            if (!expected_learned.empty()) {
                const auto expected = expected_learned.find(relative_path);
                if (expected == expected_learned.end()) {
                    throw std::runtime_error(
                        "WAV path missing from saved predictions: " + relative_path);
                }
                if (expected->second == quality.should_transcribe) {
                    ++learned_agreements;
                } else {
                    ++learned_disagreements;
                }
            }

            observe(rule, target, rule_file_admission);
            observe(learned, target, quality.should_transcribe);
            observe(hybrid, target, hybrid_admission);
            rule.asr_calls += rule_file_calls;
            learned.asr_calls += quality.should_transcribe ? 1 : 0;
            hybrid.asr_calls += hybrid_admission ? 1 : 0;
            chunk_count += chunks.size();
            audio_seconds += audio.duration_sec();
            ++examples;
        }

        const auto wall_end = std::chrono::steady_clock::now();
        const double wall_ms = std::chrono::duration<double, std::milli>(
            wall_end - wall_start).count();
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "scope=native_wav_subset\n";
        std::cout << "examples=" << examples << "\n";
        std::cout << "skipped_non_wav=" << skipped_non_wav << "\n";
        std::cout << "chunks=" << chunk_count << "\n";
        std::cout << "audio_seconds=" << audio_seconds << "\n";
        print_metrics("rule", rule, chunk_count);
        print_metrics("learned", learned, chunk_count);
        print_metrics("hybrid", hybrid, chunk_count);
        std::cout << "mean_learned_inference_us="
                  << learned_inference_us / static_cast<double>(examples) << "\n";
        if (!expected_learned.empty()) {
            std::cout << "learned_decision_agreement="
                      << learned_agreements << "/" << examples << "\n";
            std::cout << "learned_decision_disagreements="
                      << learned_disagreements << "\n";
        }
        std::cout << "gate_evaluation_wall_ms=" << wall_ms << "\n";
        std::cout << "gate_realtime_factor="
                  << (wall_ms / 1000.0) / audio_seconds << "\n";
        return examples > 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "quality policy evaluation error: " << error.what() << "\n";
        return 2;
    }
}
