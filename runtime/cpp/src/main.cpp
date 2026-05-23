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
//
// TODO(future): microphone capture path.

#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "audio/audio_io.h"
#include "asr/asr.h"
#include "chunker/chunker.h"
#include "chunker/vad.h"
#include "gate/gate.h"
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
        else {
            std::cerr << "Unknown argument: " << a << "\n";
            return false;
        }
    }

    if (out.input_path.empty()) {
        std::cerr << "ERROR: --input is required.\n";
        return false;
    }
    if (!out.gate_only && !out.vad_only && out.model_path.empty()) {
        std::cerr << "ERROR: --model is required unless --gate-only or --vad-only is set.\n";
        return false;
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
    // Chunker
    // -----------------------------------------------------------
    pipeline::ChunkerConfig chunk_cfg;
    chunk_cfg.chunk_ms = args.chunk_ms;
    chunk_cfg.hop_ms   = args.hop_ms;
    const auto chunks = pipeline::chunk_audio(audio, chunk_cfg);

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

    // -----------------------------------------------------------
    // ASR engine
    // -----------------------------------------------------------
    // ASR engine — only initialised when not in gate-only mode.
    std::unique_ptr<pipeline::AsrEngine> asr_ptr;
    if (!args.gate_only) {
        pipeline::AsrConfig asr_cfg;
        asr_cfg.model_path = args.model_path;
        asr_cfg.n_threads  = args.n_threads;
        asr_cfg.language   = args.language;
        asr_ptr = std::make_unique<pipeline::AsrEngine>(asr_cfg);
        if (!asr_ptr->ready()) {
            std::cerr << "ERROR loading model: " << asr_ptr->load_error() << "\n";
            return 1;
        }
        std::cerr << "[asr] backend_requested=" << (asr_ptr->gpu_requested() ? "cuda" : "cpu")
                  << " backend_active=" << asr_ptr->backend_mode()
                  << " cpu_fallback=" << (asr_ptr->used_cpu_fallback() ? "yes" : "no")
                  << "\n";
    } else {
        std::cerr << "[asr] gate-only mode, model not loaded\n";
    }

    // -----------------------------------------------------------
    // Main loop: gate -> ASR -> log
    // -----------------------------------------------------------
    int    n_passed     = 0;
    int    n_failed     = 0;
    int    n_borderline = 0;
    double total_infer_ms = 0.0;

    for (const auto& chunk : chunks) {
        // Apply adaptive gate adjustments from previous chunk context
        const pipeline::GateConfig effective_cfg = adaptive.adapt_gate(gate_cfg);

        // Gate
        pipeline::GateResult gate;
        if (args.gate_enabled) {
            gate = pipeline::evaluate_chunk(chunk, effective_cfg);
        } else {
            gate.decision = pipeline::GateDecision::PASS;
            gate.reason   = "gate_disabled";
            // Still compute metrics for logging even when gate is off.
            gate.metrics  = pipeline::evaluate_chunk(chunk, effective_cfg).metrics;
        }

        // Scene classification (uses pre-computed metrics; always runs)
        const pipeline::SceneResult scene = pipeline::classify(gate.metrics, scene_cfg);

        // Count
        switch (gate.decision) {
            case pipeline::GateDecision::PASS:       ++n_passed;     break;
            case pipeline::GateDecision::FAIL:       ++n_failed;     break;
            case pipeline::GateDecision::BORDERLINE: ++n_borderline; break;
        }

        // ASR: run on PASS and BORDERLINE unless gate-only or adaptive suppresses it
        pipeline::AsrResult asr_result;
        const bool run_asr =
            !args.gate_only &&
            (!args.gate_enabled ||
             gate.decision == pipeline::GateDecision::PASS ||
             gate.decision == pipeline::GateDecision::BORDERLINE) &&
            !adaptive.skip_asr(scene.label);

        if (run_asr && asr_ptr) {
            asr_result = asr_ptr->transcribe(
                chunk.samples.data(),
                static_cast<int>(chunk.samples.size()));
            total_infer_ms += asr_result.inference_ms;
        }

        // Update adaptive history with this chunk's scene label
        adaptive.push_scene(scene.label);

        logger.log_chunk(chunk, gate, asr_result, scene);
    }

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
