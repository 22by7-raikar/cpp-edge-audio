// audio_pipeline: quality admission and transcription for one complete WAV file.
//
//   audio_pipeline --input speech.wav [--model ggml-base.en.bin] [options]
//
// Without --model the file is analysed (gate, scene, learned score) but not
// transcribed. Output is one tab-separated key=value line per event on stdout;
// --report also writes a JSON report.

#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "edge_audio/asr/asr_engine.h"
#include "edge_audio/audio/audio_io.h"
#include "edge_audio/quality/adaptive_controller.h"
#include "edge_audio/quality/admission_policy.h"
#include "edge_audio/quality/file_features.h"
#include "edge_audio/quality/rule_gate.h"
#include "edge_audio/quality/scene_classifier.h"
#include "edge_audio/report/run_logger.h"
#include "edge_audio/segmentation/fixed_chunker.h"
#include "edge_audio/segmentation/vad.h"

namespace ea = edge_audio;

namespace {

constexpr int kAsrSampleRate = 16000;
constexpr int kTrainingChunkMs = 5000;

enum class Segmentation { FIXED, VAD, VAD_PACKED };

struct Options {
    std::string input_path;
    std::string model_path;   // empty: analysis only
    std::string report_path;  // empty: no JSON report
    std::string language = "en";
    Segmentation segmentation = Segmentation::FIXED;
    int chunk_ms = kTrainingChunkMs;
    int threads = 4;
    bool gate_enabled = true;
    bool adaptive_enabled = true;
    bool debug_features = false;
    ea::QualityPolicy policy = ea::QualityPolicy::RULE;
    double threshold = ea::kQualityDefaultThreshold;
};

const char* segmentation_str(Segmentation s) {
    switch (s) {
        case Segmentation::FIXED:
            return "fixed";
        case Segmentation::VAD:
            return "vad";
        case Segmentation::VAD_PACKED:
            return "vad-packed";
    }
    return "fixed";
}

void print_usage(const char* program) {
    std::cerr << "usage: " << program << " --input <wav> [options]\n\n"
              << "  --model <path>            whisper.cpp model; omit to analyse without ASR\n"
              << "  --policy <name>           rule | learned | hybrid (default: rule)\n"
              << "  --threshold <p>           learned probability threshold in [0,1] (default: "
              << ea::kQualityDefaultThreshold << ")\n"
              << "  --segmentation <mode>     fixed | vad | vad-packed (default: fixed)\n"
              << "  --chunk-ms <n>            fixed window length (default: 5000)\n"
              << "  --no-gate                 send every window to ASR (rule policy only)\n"
              << "  --no-adapt                disable the scene-adaptive controller\n"
              << "  --threads <n>             ASR threads (default: 4)\n"
              << "  --language <code>         whisper language hint (default: en)\n"
              << "  --report <path>           also write a JSON report\n"
              << "  --debug-features          include the 27-feature vector in the output\n";
}

int parse_int(const std::string& option, const std::string& value) {
    std::size_t used = 0;
    const int parsed = std::stoi(value, &used);
    if (used != value.size()) throw std::invalid_argument(option + " expects an integer");
    return parsed;
}

Options parse_options(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto value = [&]() -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument(arg + " requires a value");
            return argv[++i];
        };
        if (arg == "--input")
            o.input_path = value();
        else if (arg == "--model")
            o.model_path = value();
        else if (arg == "--report")
            o.report_path = value();
        else if (arg == "--language")
            o.language = value();
        else if (arg == "--policy")
            o.policy = ea::parse_quality_policy(value());
        else if (arg == "--threshold")
            o.threshold = ea::parse_quality_threshold(value());
        else if (arg == "--chunk-ms")
            o.chunk_ms = parse_int(arg, value());
        else if (arg == "--threads")
            o.threads = parse_int(arg, value());
        else if (arg == "--no-gate")
            o.gate_enabled = false;
        else if (arg == "--no-adapt")
            o.adaptive_enabled = false;
        else if (arg == "--debug-features")
            o.debug_features = true;
        else if (arg == "--segmentation") {
            const std::string mode = value();
            if (mode == "fixed")
                o.segmentation = Segmentation::FIXED;
            else if (mode == "vad")
                o.segmentation = Segmentation::VAD;
            else if (mode == "vad-packed")
                o.segmentation = Segmentation::VAD_PACKED;
            else
                throw std::invalid_argument("--segmentation must be fixed, vad or vad-packed");
        } else {
            throw std::invalid_argument("unknown option: " + arg);
        }
    }

    if (o.input_path.empty()) throw std::invalid_argument("--input is required");
    if (o.chunk_ms <= 0) throw std::invalid_argument("--chunk-ms must be positive");
    if (o.threads <= 0) throw std::invalid_argument("--threads must be positive");
    if (o.policy != ea::QualityPolicy::RULE) {
        // The learned model only means something on the inputs it was trained on.
        if (o.segmentation != Segmentation::FIXED || o.chunk_ms != kTrainingChunkMs) {
            throw std::invalid_argument(
                "learned and hybrid policies require --segmentation fixed --chunk-ms 5000");
        }
        if (!o.gate_enabled) {
            throw std::invalid_argument("learned and hybrid policies require the rule gate");
        }
    }
    return o;
}

std::vector<ea::Chunk> build_chunks(const ea::AudioBuffer& audio, const Options& o) {
    if (o.segmentation == Segmentation::FIXED) {
        ea::ChunkerConfig cfg;
        cfg.chunk_ms = o.chunk_ms;
        return ea::chunk_audio(audio, cfg);
    }

    const auto speech = ea::run_vad(audio.samples.data(), static_cast<int>(audio.samples.size()),
                                    audio.sample_rate);
    const auto windows = o.segmentation == Segmentation::VAD_PACKED
                             ? ea::pack_vad_segments(speech, audio.duration_sec())
                             : speech;
    std::cerr << "[segmentation] " << speech.size() << " speech regions -> " << windows.size()
              << " ASR windows\n";

    std::vector<ea::Chunk> chunks;
    chunks.reserve(windows.size());
    const auto total = static_cast<long>(audio.samples.size());
    for (std::size_t i = 0; i < windows.size(); ++i) {
        const long begin =
            std::max(0L, static_cast<long>(windows[i].start_sec * audio.sample_rate));
        const long end = std::min(total, static_cast<long>(windows[i].end_sec * audio.sample_rate));
        ea::Chunk chunk;
        chunk.sample_rate = audio.sample_rate;
        chunk.index = static_cast<int>(i);
        chunk.start_sec = windows[i].start_sec;
        chunk.end_sec = windows[i].end_sec;
        if (begin < end) {
            chunk.samples.assign(audio.samples.begin() + begin, audio.samples.begin() + end);
        }
        chunks.push_back(std::move(chunk));
    }
    return chunks;
}

struct Counts {
    int passed = 0;
    int failed = 0;
    int borderline = 0;
    double asr_ms = 0.0;

    void add(ea::GateDecision d) {
        if (d == ea::GateDecision::PASS)
            ++passed;
        else if (d == ea::GateDecision::FAIL)
            ++failed;
        else
            ++borderline;
    }
};

// Lazily loaded so analysis-only and rejected runs never pay the model load.
class LazyAsr {
public:
    explicit LazyAsr(const Options& o) : options_(o) {}

    bool enabled() const { return !options_.model_path.empty(); }

    ea::AsrEngine& engine() {
        if (!engine_) {
            ea::AsrConfig cfg;
            cfg.model_path = options_.model_path;
            cfg.n_threads = options_.threads;
            cfg.language = options_.language;
            engine_ = std::make_unique<ea::AsrEngine>(cfg);
            if (!engine_->ready()) throw std::runtime_error(engine_->load_error());
            std::cerr << "[asr] backend=" << engine_->backend_mode()
                      << " cpu_fallback=" << (engine_->used_cpu_fallback() ? "yes" : "no") << '\n';
        }
        return *engine_;
    }

private:
    const Options& options_;
    std::unique_ptr<ea::AsrEngine> engine_;
};

// RULE: every window is gated with the adaptive configuration, and each admitted
// window is transcribed on its own.
void run_rule_policy(const std::vector<ea::Chunk>& chunks, const Options& o, LazyAsr& asr,
                     ea::RunLogger& logger, Counts& counts, ea::QualitySummaryRecord& summary) {
    const ea::GateConfig base_gate;
    const ea::SceneConfig scene_cfg;
    ea::AdaptiveConfig adapt_cfg;
    adapt_cfg.enabled = o.adaptive_enabled;
    ea::AdaptiveController adaptive(adapt_cfg);

    bool any_admitted = false;
    bool asr_ran = false;
    for (const auto& chunk : chunks) {
        ea::GateResult gate = ea::evaluate_chunk(chunk, adaptive.adapt_gate(base_gate));
        if (!o.gate_enabled) {
            gate.decision = ea::GateDecision::PASS;
            gate.reason = "gate_disabled";
        }
        const ea::SceneResult scene = ea::classify(gate.metrics, scene_cfg);
        counts.add(gate.decision);

        const bool admitted =
            gate.decision != ea::GateDecision::FAIL && !adaptive.skip_asr(scene.label);
        any_admitted = any_admitted || admitted;

        ea::AsrResult result;
        if (admitted && asr.enabled()) {
            result = asr.engine().transcribe(chunk.samples.data(),
                                             static_cast<int>(chunk.samples.size()));
            counts.asr_ms += result.inference_ms;
            asr_ran = true;
        }
        adaptive.push_scene(scene.label);
        logger.log_chunk(chunk, gate, result, scene);
    }

    const auto admission =
        ea::decide_quality_admission(ea::QualityPolicy::RULE, any_admitted, nullptr);
    summary.rule_summary = any_admitted;
    summary.final_admission = admission.final_should_transcribe;
    summary.rejection_reason = admission.rejection_reason;
    summary.asr_ran = asr_ran;
    summary.asr_inference_ms = counts.asr_ms;
}

// LEARNED / HYBRID: the 27-feature vector is built from the fixed training-time
// gate, scored once, and the whole file is transcribed at most once.
void run_learned_policy(const std::vector<ea::Chunk>& chunks, const ea::AudioBuffer& audio,
                        const Options& o, LazyAsr& asr, ea::RunLogger& logger, Counts& counts,
                        ea::QualitySummaryRecord& summary) {
    const ea::GateConfig base_gate;
    const ea::SceneConfig scene_cfg;
    ea::AdaptiveConfig adapt_cfg;
    adapt_cfg.enabled = o.adaptive_enabled;
    ea::AdaptiveController adaptive(adapt_cfg);
    ea::QualityFeatureAggregator aggregator;

    bool rule_admits = false;
    for (const auto& chunk : chunks) {
        const ea::GateResult gate = ea::evaluate_chunk(chunk, base_gate);
        const ea::SceneResult scene = ea::classify(gate.metrics, scene_cfg);
        aggregator.add(gate.metrics, gate.decision);
        counts.add(gate.decision);

        // The rule half of HYBRID uses the same adaptive scheduling as RULE.
        const ea::GateResult rule_gate =
            adaptive.dominant_scene() == ea::SceneLabel::NOISE
                ? ea::evaluate_chunk(chunk, adaptive.adapt_gate(base_gate))
                : gate;
        rule_admits = rule_admits || (rule_gate.decision != ea::GateDecision::FAIL &&
                                      !adaptive.skip_asr(scene.label));
        adaptive.push_scene(scene.label);
        logger.log_chunk(chunk, gate, ea::AsrResult{}, scene);
    }
    if (aggregator.empty()) throw std::runtime_error("input produced no analysis windows");

    const auto features = aggregator.features();
    const auto started = std::chrono::steady_clock::now();
    const ea::QualityPrediction prediction = ea::predict_quality(features, o.threshold);
    const double inference_us =
        std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - started)
            .count();
    const auto admission = ea::decide_quality_admission(o.policy, rule_admits, &prediction);

    const ea::AsrRun run = ea::transcribe_if_admitted(
        admission.final_should_transcribe, asr.enabled(), audio.samples.data(),
        audio.samples.size(),
        [&asr](const float* samples, int n) { return asr.engine().transcribe(samples, n); });
    counts.asr_ms = run.result.inference_ms;

    summary.features_available = true;
    summary.features = features;
    summary.learned_evaluated = true;
    summary.learned_raw_score = prediction.raw_score;
    summary.learned_probability = prediction.probability;
    summary.learned_inference_us = inference_us;
    summary.learned_decision = prediction.should_transcribe;
    summary.rule_summary = rule_admits;
    summary.final_admission = admission.final_should_transcribe;
    summary.rejection_reason = admission.rejection_reason;
    summary.asr_ran = run.ran;
    summary.asr_inference_ms = run.result.inference_ms;
    summary.transcript = run.result.text;
    summary.asr_error = run.result.error;
}

int run(const Options& o) {
    ea::AudioBuffer audio;
    std::string error;
    if (!ea::load_wav(o.input_path, audio, error)) throw std::runtime_error(error);
    if (audio.sample_rate != kAsrSampleRate) audio = ea::resample(audio, kAsrSampleRate);

    ea::RunLogger logger;
    logger.set_json_path(o.report_path);
    ea::RunConfig run_cfg;
    run_cfg.input_path = o.input_path;
    run_cfg.model_path = o.model_path;
    run_cfg.segmentation = segmentation_str(o.segmentation);
    run_cfg.chunk_ms = o.chunk_ms;
    run_cfg.n_threads = o.threads;
    run_cfg.gate_enabled = o.gate_enabled;
    run_cfg.quality_policy = ea::quality_policy_str(o.policy);
    run_cfg.quality_threshold = o.threshold;
    logger.log_run_start(run_cfg);

    const std::vector<ea::Chunk> chunks = build_chunks(audio, o);
    LazyAsr asr(o);
    Counts counts;
    ea::QualitySummaryRecord summary;
    summary.policy = ea::quality_policy_str(o.policy);
    summary.chunk_count = static_cast<int>(chunks.size());
    summary.schema_version = std::string(ea::quality_schema_version());
    summary.threshold = o.threshold;
    summary.debug_features = o.debug_features;

    if (o.policy == ea::QualityPolicy::RULE) {
        run_rule_policy(chunks, o, asr, logger, counts, summary);
    } else {
        run_learned_policy(chunks, audio, o, asr, logger, counts, summary);
    }

    logger.log_quality_summary(summary);
    const bool report_ok =
        logger.log_run_end(static_cast<int>(chunks.size()), counts.passed, counts.failed,
                           counts.borderline, audio.duration_sec(), counts.asr_ms);
    return report_ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    Options options;
    try {
        options = parse_options(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n\n";
        print_usage(argv[0]);
        return 2;
    }
    try {
        return run(options);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
