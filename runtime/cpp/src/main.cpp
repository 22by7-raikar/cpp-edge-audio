// main.cpp
// Command-line entry point for the audio ML pipeline.
//
// Usage:
//   audio_pipeline --input <wav> --model <model.bin> [options]
//
// Options:
//   --input         / -i   <path>    Input WAV file (any sample rate; resampled to 16kHz)
//   --model         / -m   <path>    whisper.cpp model file (.bin)
//   --chunk-ms             <int>     Chunk size in ms (default: 5000)
//   --hop-ms               <int>     Chunk hop in ms, 0 = no overlap (default: 0)
//   --threads       / -t   <int>     Inference thread count (default: 4)
//   --no-gate                        Disable gate; all chunks sent to ASR
//   --no-adapt                       Disable adaptive controller
//   --rms-min              <float>   Min RMS for PASS (default: 0.003)
//   --max-silence          <float>   Max silence ratio for PASS (default: 0.90)
//   --max-clip             <float>   Max clipping ratio for PASS (default: 0.05)
//   --max-flatness         <float>   Max spectral flatness mean for PASS (default: 0.90)
//   --frame-size           <int>     FFT frame size, power of 2 (default: 512)
//   --log           / -l   <path>    Write TSV log to this file in addition to stdout
//   --bench-json    / -j   <path>    Write JSON benchmark file (per-chunk + summary)
//   --language             <str>     Language hint for Whisper (default: en)
//   --gate-only                      Run gate + scene only; skip ASR (--model not required)
//   --vad-only                       Run DSP VAD segmentation only; print JSON to stdout
//   --vad-asr                        Use VAD segmentation instead of fixed-window chunking; gate + ASR still run
//   --vad-asr-packed                 VAD segmentation with padding + merging to wider ASR windows (fewer CUDA calls)
//   --quality-policy       <str>     rule, learned, or hybrid (default: rule)
//   --quality-threshold    <float>   Learned probability threshold in [0,1] (default: 0.3)
//   --quality-debug-features         Log the ordered 27-feature vector
//
// TODO(future): microphone capture path.

#include <chrono>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "audio/audio_io.h"
#include "asr/asr.h"
#include "chunker/chunker.h"
#include "chunker/vad.h"
#include "gate/gate.h"
#include "gate/quality_aggregator.h"
#include "gate/quality_model.h"
#include "gate/quality_policy.h"
#include "logging/logger.h"
#include "scene/scene.h"
#include "scene/adaptive.h"

namespace {

void print_usage(const char* prog) {
    std::cerr
        << "Usage: " << prog
        << " --input <wav> --model <model.bin>"
           " [--chunk-ms 5000] [--hop-ms 0] [--threads 4]"
           " [--no-gate] [--no-adapt] [--rms-min 0.003] [--max-silence 0.90]"
           " [--max-clip 0.05] [--max-flatness 0.90] [--frame-size 512]"
           " [--quality-policy rule|learned|hybrid] [--quality-threshold 0.3]"
           " [--quality-debug-features]"
           " [--log <tsv>] [--bench-json <json>] [--language en]\n";
}

struct CliArgs {
    std::string input_path;
    std::string model_path;
    std::string log_path;
    std::string json_path;
    std::string language       = "en";
    int         chunk_ms       = 5000;
    int         hop_ms         = 0;
    int         n_threads      = 4;
    int         frame_size     = 512;
    double      rms_min        = 0.003;
    double      max_silence    = 0.90;
    double      max_clip       = 0.05;
    double      max_flatness   = 0.90;
    bool        gate_enabled     = true;
    bool        adaptive_enabled  = true;
    bool        gate_only         = false;  // skip ASR, model not required
    bool        vad_only          = false;  // run VAD only, print JSON to stdout
    bool        vad_asr           = false;  // VAD segmentation instead of fixed-window chunking
    bool        vad_asr_packed     = false;  // VAD + padding/merge packing before ASR
    pipeline::QualityPolicy quality_policy = pipeline::QualityPolicy::RULE;
    double      quality_threshold = pipeline::kQualityDefaultThreshold;
    bool        quality_debug_features = false;
};

bool parse_args(int argc, char** argv, CliArgs& out) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "ERROR: " << a << " requires an argument.\n";
                return "";
            }
            return argv[++i];
        };

        if      (a == "--input"       || a == "-i") out.input_path  = next();
        else if (a == "--model"       || a == "-m") out.model_path  = next();
        else if (a == "--log"         || a == "-l") out.log_path    = next();
        else if (a == "--bench-json"  || a == "-j") out.json_path   = next();
        else if (a == "--language")                 out.language    = next();
        else if (a == "--chunk-ms")                 out.chunk_ms    = std::stoi(next());
        else if (a == "--hop-ms")                   out.hop_ms      = std::stoi(next());
        else if (a == "--threads"     || a == "-t") out.n_threads   = std::stoi(next());
        else if (a == "--frame-size")               out.frame_size  = std::stoi(next());
        else if (a == "--rms-min")                  out.rms_min     = std::stod(next());
        else if (a == "--max-silence")              out.max_silence = std::stod(next());
        else if (a == "--max-clip")                 out.max_clip    = std::stod(next());
        else if (a == "--max-flatness")             out.max_flatness = std::stod(next());
        else if (a == "--no-gate")                  out.gate_enabled = false;
        else if (a == "--no-adapt")                 out.adaptive_enabled = false;
        else if (a == "--gate-only")                out.gate_only = true;
        else if (a == "--vad-only")                 out.vad_only  = true;
        else if (a == "--vad-asr")                  out.vad_asr        = true;
        else if (a == "--vad-asr-packed")             out.vad_asr_packed = true;
        else if (a == "--quality-policy") {
            try {
                out.quality_policy = pipeline::parse_quality_policy(next());
            } catch (const std::invalid_argument& error) {
                std::cerr << "ERROR: " << error.what() << "\n";
                return false;
            }
        }
        else if (a == "--quality-threshold") {
            try {
                out.quality_threshold = pipeline::parse_quality_threshold(next());
            } catch (const std::invalid_argument& error) {
                std::cerr << "ERROR: " << error.what() << "\n";
                return false;
            }
        }
        else if (a == "--quality-debug-features") out.quality_debug_features = true;
        else {
            std::cerr << "Unknown argument: " << a << "\n";
            return false;
        }
    }

    if (out.input_path.empty()) {
        std::cerr << "ERROR: --input is required.\n";
        return false;
    }
    if (!out.gate_only && !out.vad_only && !out.vad_asr && !out.vad_asr_packed && out.model_path.empty()) {
        std::cerr << "ERROR: --model is required unless --gate-only or --vad-only is set.\n";
        return false;
    }
    if (out.quality_policy != pipeline::QualityPolicy::RULE) {
        if (out.vad_only || out.vad_asr || out.vad_asr_packed) {
            std::cerr << "ERROR: learned and hybrid quality policies require a complete WAV boundary; VAD modes are unsupported.\n";
            return false;
        }
        if (out.chunk_ms != 5000 || out.hop_ms != 0) {
            std::cerr << "ERROR: learned and hybrid quality policies require non-overlapping 5000 ms analysis chunks.\n";
            return false;
        }
        if (!out.gate_enabled) {
            std::cerr << "ERROR: learned and hybrid quality policies require gate decisions for the 27-feature schema.\n";
            return false;
        }
        if (out.frame_size != 512 || out.rms_min != 0.003 ||
            out.max_silence != 0.90 || out.max_clip != 0.05 ||
            out.max_flatness != 0.90) {
            std::cerr << "ERROR: learned and hybrid quality policies require the training-time DSP and rule-gate configuration.\n";
            return false;
        }
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    CliArgs args;
    if (!parse_args(argc, argv, args)) {
        print_usage(argv[0]);
        return 1;
    }

    std::cerr << "[quality] policy="
              << pipeline::quality_policy_str(args.quality_policy)
              << " threshold=" << args.quality_threshold;
    if (args.quality_policy != pipeline::QualityPolicy::RULE) {
        std::cerr << " boundary=complete_wav analysis_chunks=5000ms_non_overlapping";
    }
    std::cerr << "\n";

    // -----------------------------------------------------------
    // Logger
    // -----------------------------------------------------------
    pipeline::Logger logger;
    if (!args.log_path.empty()) {
        if (!logger.open(args.log_path)) {
            std::cerr << "WARNING: could not open TSV log: " << args.log_path << "\n";
        }
    }
    if (!args.json_path.empty()) {
        logger.set_json_path(args.json_path);
    }

    pipeline::RunConfig run_cfg;
    run_cfg.input_path   = args.input_path;
    run_cfg.model_path   = args.model_path;
    run_cfg.chunk_ms     = args.chunk_ms;
    run_cfg.hop_ms       = args.hop_ms;
    run_cfg.n_threads    = args.n_threads;
    run_cfg.gate_enabled = args.gate_enabled;
    run_cfg.quality_policy = pipeline::quality_policy_str(args.quality_policy);
    run_cfg.quality_threshold = args.quality_threshold;
    logger.log_run_start(run_cfg);

    // -----------------------------------------------------------
    // Load audio
    // -----------------------------------------------------------
    pipeline::AudioBuffer audio;
    std::string load_err;
    if (!pipeline::load_wav(args.input_path, audio, load_err)) {
        std::cerr << "ERROR loading audio: " << load_err << "\n";
        return 1;
    }

    // Whisper requires 16 kHz; resample if needed.
    const int WHISPER_SR = 16000;
    if (audio.sample_rate != WHISPER_SR) {
        std::cerr << "Resampling from " << audio.sample_rate
                  << " Hz to " << WHISPER_SR << " Hz...\n";
        audio = pipeline::resample(audio, WHISPER_SR);
    }

    // -----------------------------------------------------------
    // --vad-only: run VAD, print JSON, exit
    // -----------------------------------------------------------
    if (args.vad_only) {
        pipeline::VadConfig vad_cfg;
        const auto segs = pipeline::run_vad(
            audio.samples.data(),
            static_cast<int>(audio.samples.size()),
            audio.sample_rate,
            vad_cfg);

        double speech_dur = 0.0;
        for (const auto& s : segs) speech_dur += s.duration_sec();

        const double total_dur = audio.duration_sec();
        const double speech_frac = (total_dur > 0.0) ? speech_dur / total_dur : 0.0;

        std::printf("{\n");
        std::printf("  \"input\": \"%s\",\n", args.input_path.c_str());
        std::printf("  \"sample_rate\": %d,\n", audio.sample_rate);
        std::printf("  \"total_duration_sec\": %.4f,\n", total_dur);
        std::printf("  \"n_segments\": %zu,\n", segs.size());
        std::printf("  \"speech_duration_sec\": %.4f,\n", speech_dur);
        std::printf("  \"speech_fraction\": %.4f,\n", speech_frac);
        std::printf("  \"vad_segments\": [\n");
        for (size_t i = 0; i < segs.size(); ++i) {
            const auto& s = segs[i];
            std::printf(
                "    {\"start_sec\": %.4f, \"end_sec\": %.4f,"
                " \"duration_sec\": %.4f, \"speech_ratio\": %.4f,"
                " \"frame_count\": %d}%s\n",
                s.start_sec, s.end_sec, s.duration_sec(),
                s.speech_ratio, s.frame_count,
                (i + 1 < segs.size()) ? "," : "");
        }
        std::printf("  ]\n}\n");
        return 0;
    }

    // -----------------------------------------------------------
    // Build chunk list: fixed-window or VAD-based
    // -----------------------------------------------------------
    std::vector<pipeline::Chunk> chunks;
    if (args.vad_asr || args.vad_asr_packed) {
        pipeline::VadConfig vad_cfg;
        const auto raw_segs = pipeline::run_vad(
            audio.samples.data(),
            static_cast<int>(audio.samples.size()),
            audio.sample_rate,
            vad_cfg);

        // Optionally pack raw segments into wider ASR windows.
        const std::vector<pipeline::VadSegment>& segs =
            args.vad_asr_packed
            ? pipeline::pack_vad_segments(
                  raw_segs,
                  audio.duration_sec())
            : raw_segs;

        std::cerr << (args.vad_asr_packed ? "[vad-packed] " : "[vad-asr] ")
                  << raw_segs.size() << " raw segments";
        if (args.vad_asr_packed)
            std::cerr << " -> " << segs.size() << " packed windows";
        std::cerr << "\n";

        chunks.reserve(segs.size());
        for (size_t i = 0; i < segs.size(); ++i) {
            const auto& seg = segs[i];
            const int s_idx = std::max(0,
                static_cast<int>(seg.start_sec * audio.sample_rate));
            const int e_idx = std::min(
                static_cast<int>(audio.samples.size()),
                static_cast<int>(seg.end_sec   * audio.sample_rate));
            pipeline::Chunk c;
            c.sample_rate = audio.sample_rate;
            c.index       = static_cast<int>(i);
            c.start_sec   = seg.start_sec;
            c.end_sec     = seg.end_sec;
            if (s_idx < e_idx)
                c.samples.assign(audio.samples.begin() + s_idx,
                                 audio.samples.begin() + e_idx);
            chunks.push_back(std::move(c));
        }
    } else {
        pipeline::ChunkerConfig chunk_cfg;
        chunk_cfg.chunk_ms = args.chunk_ms;
        chunk_cfg.hop_ms   = args.hop_ms;
        chunks = pipeline::chunk_audio(audio, chunk_cfg);
    }

    // -----------------------------------------------------------
    // Gate config
    // -----------------------------------------------------------
    pipeline::GateConfig gate_cfg;
    gate_cfg.rms_min               = args.rms_min;
    gate_cfg.max_silence_ratio     = args.max_silence;
    gate_cfg.max_clipping_ratio    = args.max_clip;
    gate_cfg.spectral_flatness_max = args.max_flatness;
    gate_cfg.frame.frame_size      = args.frame_size;
    gate_cfg.frame.hop_size        = args.frame_size / 2;  // 50% overlap

    // -----------------------------------------------------------
    // Scene classifier + adaptive controller
    // -----------------------------------------------------------
    pipeline::SceneConfig    scene_cfg;   // default thresholds
    pipeline::AdaptiveConfig adapt_cfg;
    adapt_cfg.enabled = args.adaptive_enabled;
    pipeline::AdaptiveController adaptive(adapt_cfg);

    // ASR model loading is deferred until the selected policy has admitted
    // audio. Rule mode still initializes it at the same point as before.
    std::unique_ptr<pipeline::AsrEngine> asr_ptr;
    auto initialize_asr = [&]() -> bool {
        if (asr_ptr) return true;
        pipeline::AsrConfig asr_cfg;
        asr_cfg.model_path = args.model_path;
        asr_cfg.n_threads  = args.n_threads;
        asr_cfg.language   = args.language;
        asr_ptr = std::make_unique<pipeline::AsrEngine>(asr_cfg);
        if (!asr_ptr->ready()) {
            std::cerr << "ERROR loading model: " << asr_ptr->load_error() << "\n";
            return false;
        }
        std::cerr << "[asr] backend_requested=" << (asr_ptr->gpu_requested() ? "cuda" : "cpu")
                  << " backend_active=" << asr_ptr->backend_mode()
                  << " cpu_fallback=" << (asr_ptr->used_cpu_fallback() ? "yes" : "no")
                  << "\n";
        return true;
    };
    if (args.gate_only) {
        std::cerr << "[asr] gate-only mode, model not loaded\n";
    }

    // -----------------------------------------------------------
    // Main loop: gate -> ASR -> log
    // -----------------------------------------------------------
    int    n_passed     = 0;
    int    n_failed     = 0;
    int    n_borderline = 0;
    double total_infer_ms = 0.0;
    pipeline::QualitySummaryRecord quality_summary;
    quality_summary.policy = pipeline::quality_policy_str(args.quality_policy);
    quality_summary.chunk_count = static_cast<int>(chunks.size());
    quality_summary.schema_version = std::string(pipeline::quality_schema_version());
    quality_summary.threshold = args.quality_threshold;
    quality_summary.debug_features = args.quality_debug_features;

    if (args.quality_policy == pipeline::QualityPolicy::RULE) {
        if (!args.gate_only && !initialize_asr()) return 1;

        bool rule_should_transcribe = false;
        bool asr_ran = false;
        for (const auto& chunk : chunks) {
            // Apply adaptive gate adjustments from previous chunk context.
            const pipeline::GateConfig effective_cfg = adaptive.adapt_gate(gate_cfg);

            pipeline::GateResult gate;
            if (args.gate_enabled) {
                gate = pipeline::evaluate_chunk(chunk, effective_cfg);
            } else {
                gate.decision = pipeline::GateDecision::PASS;
                gate.reason   = "gate_disabled";
                gate.metrics  = pipeline::evaluate_chunk(chunk, effective_cfg).metrics;
            }

            const pipeline::SceneResult scene =
                pipeline::classify(gate.metrics, scene_cfg);
            switch (gate.decision) {
                case pipeline::GateDecision::PASS:       ++n_passed;     break;
                case pipeline::GateDecision::FAIL:       ++n_failed;     break;
                case pipeline::GateDecision::BORDERLINE: ++n_borderline; break;
            }

            const bool rule_admits_chunk =
                (!args.gate_enabled ||
                 gate.decision == pipeline::GateDecision::PASS ||
                 gate.decision == pipeline::GateDecision::BORDERLINE) &&
                !adaptive.skip_asr(scene.label);
            rule_should_transcribe =
                rule_should_transcribe || rule_admits_chunk;
            const bool run_asr = !args.gate_only && rule_admits_chunk;

            pipeline::AsrResult asr_result;
            if (run_asr && asr_ptr) {
                asr_ran = true;
                asr_result = asr_ptr->transcribe(
                    chunk.samples.data(),
                    static_cast<int>(chunk.samples.size()));
                total_infer_ms += asr_result.inference_ms;
            }

            adaptive.push_scene(scene.label);
            logger.log_chunk(chunk, gate, asr_result, scene);
        }

        const pipeline::QualityAdmission admission =
            pipeline::decide_quality_admission(
                pipeline::QualityPolicy::RULE,
                rule_should_transcribe,
                nullptr);
        quality_summary.rule_summary = rule_should_transcribe;
        quality_summary.final_admission = admission.final_should_transcribe;
        quality_summary.asr_ran = asr_ran;
        quality_summary.rejection_reason = admission.rejection_reason;
        quality_summary.asr_inference_ms = total_infer_ms;
    } else {
        pipeline::QualityFeatureAggregator aggregator;
        bool rule_should_transcribe = false;

        // Learned-model analysis always uses the immutable training boundary:
        // the complete file split into non-overlapping five-second chunks.
        // Scene/adaptive routing does not alter the learned feature vector.
        for (const auto& chunk : chunks) {
            const pipeline::GateResult gate =
                pipeline::evaluate_chunk(chunk, gate_cfg);
            const pipeline::SceneResult scene =
                pipeline::classify(gate.metrics, scene_cfg);
            aggregator.add(gate.metrics, gate.decision);

            // Reproduce the current rule scheduler for the file-level rule
            // summary. The learned vector still uses the fixed base gate above.
            pipeline::GateResult rule_gate = gate;
            if (adaptive.dominant_scene() == pipeline::SceneLabel::NOISE) {
                rule_gate = pipeline::evaluate_chunk(
                    chunk, adaptive.adapt_gate(gate_cfg));
            }
            const bool rule_admits_chunk =
                (rule_gate.decision == pipeline::GateDecision::PASS ||
                 rule_gate.decision == pipeline::GateDecision::BORDERLINE) &&
                !adaptive.skip_asr(scene.label);
            rule_should_transcribe =
                rule_should_transcribe || rule_admits_chunk;

            switch (gate.decision) {
                case pipeline::GateDecision::PASS:
                    ++n_passed;
                    break;
                case pipeline::GateDecision::BORDERLINE:
                    ++n_borderline;
                    break;
                case pipeline::GateDecision::FAIL:
                    ++n_failed;
                    break;
            }
            adaptive.push_scene(scene.label);
            logger.log_chunk(chunk, gate, pipeline::AsrResult{}, scene);
        }

        if (aggregator.empty()) {
            std::cerr << "ERROR: complete WAV produced no quality-analysis chunks\n";
            return 1;
        }

        const auto features = aggregator.features();
        const auto quality_start = std::chrono::steady_clock::now();
        const pipeline::QualityPrediction prediction =
            pipeline::predict_quality(features, args.quality_threshold);
        const auto quality_end = std::chrono::steady_clock::now();
        const double learned_inference_us =
            std::chrono::duration<double, std::micro>(
                quality_end - quality_start).count();
        const pipeline::QualityAdmission admission =
            pipeline::decide_quality_admission(
                args.quality_policy,
                rule_should_transcribe,
                &prediction);

        if (admission.final_should_transcribe && !args.gate_only &&
            !initialize_asr()) {
            return 1;
        }
        const pipeline::FileAsrExecution execution =
            pipeline::transcribe_admitted_file_once(
                admission.final_should_transcribe,
                !args.gate_only,
                audio.samples.data(),
                audio.samples.size(),
                [&](const float* samples, int sample_count) {
                    return asr_ptr->transcribe(samples, sample_count);
                });
        if (!admission.final_should_transcribe && !args.gate_only) {
            std::cerr << "[asr] skipped: " << admission.rejection_reason << "\n";
        }
        total_infer_ms = execution.result.inference_ms;

        quality_summary.features_available = true;
        quality_summary.features = features;
        quality_summary.learned_evaluated = true;
        quality_summary.learned_raw_score = prediction.raw_score;
        quality_summary.learned_probability = prediction.probability;
        quality_summary.learned_inference_us = learned_inference_us;
        quality_summary.learned_decision = prediction.should_transcribe;
        quality_summary.rule_summary = rule_should_transcribe;
        quality_summary.final_admission = admission.final_should_transcribe;
        quality_summary.asr_ran = execution.ran;
        quality_summary.rejection_reason = admission.rejection_reason;
        quality_summary.asr_inference_ms = execution.result.inference_ms;
        quality_summary.transcript = execution.result.text;
        quality_summary.asr_error = execution.result.error;
    }

    logger.log_quality_summary(quality_summary);

    // -----------------------------------------------------------
    // Summary
    // -----------------------------------------------------------
    logger.log_run_end(
        static_cast<int>(chunks.size()),
        n_passed,
        n_failed,
        n_borderline,
        audio.duration_sec(),
        total_infer_ms);

    return 0;
}
