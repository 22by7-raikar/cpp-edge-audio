// asr.cpp
// Wraps whisper.cpp C API (whisper_context / whisper_full).
// Whisper requires mono float32 PCM at exactly 16000 Hz.
// Resampling is the caller's responsibility; see TODO in main.cpp.

#include "asr.h"
#include "whisper.h"

#include <chrono>
#include <cstring>
#include <sstream>

namespace pipeline {

AsrEngine::AsrEngine(const AsrConfig& cfg) : cfg_(cfg) {
    whisper_context_params cparams = whisper_context_default_params();
#ifdef GGML_CUDA
    gpu_requested_ = true;
    cparams.use_gpu = true;
    cparams.gpu_device = 0;
#else
    gpu_requested_ = false;
    cparams.use_gpu = false;
#endif

    ctx_ = whisper_init_from_file_with_params(cfg.model_path.c_str(), cparams);
    if (!ctx_ && gpu_requested_) {
        // CUDA requested but unavailable at runtime/device selection failed.
        cpu_fallback_ = true;
        cparams.use_gpu = false;
        cparams.gpu_device = 0;
        ctx_ = whisper_init_from_file_with_params(cfg.model_path.c_str(), cparams);
    }

    if (!ctx_) {
        load_error_ = "whisper_init_from_file_with_params failed: " + cfg.model_path;
        backend_mode_ = "unavailable";
        gpu_enabled_ = false;
        return;
    }

    gpu_enabled_ = gpu_requested_ && !cpu_fallback_;
    backend_mode_ = gpu_enabled_ ? "cuda" : "cpu";
}

AsrEngine::~AsrEngine() {
    if (ctx_) {
        whisper_free(ctx_);
        ctx_ = nullptr;
    }
}

AsrResult AsrEngine::transcribe(const float* samples, int n_samples) {
    AsrResult result;

    if (!ctx_) {
        result.error = load_error_;
        return result;
    }

    if (n_samples <= 0) {
        result.error = "zero samples";
        return result;
    }

    whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    params.n_threads       = cfg_.n_threads;
    params.translate       = cfg_.translate;
    params.no_timestamps   = cfg_.no_timestamps;
    params.single_segment  = false;
    params.print_special   = false;
    params.print_progress  = false;
    params.print_realtime  = false;
    params.print_timestamps = false;

    if (!cfg_.language.empty() && cfg_.language != "auto") {
        params.language = cfg_.language.c_str();
    }

    const auto t0 = std::chrono::steady_clock::now();
    const int rc  = whisper_full(ctx_, params, samples, n_samples);
    const auto t1 = std::chrono::steady_clock::now();

    result.inference_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (rc != 0) {
        result.error = "whisper_full returned " + std::to_string(rc);
        return result;
    }

    std::ostringstream oss;
    const int nseg = whisper_full_n_segments(ctx_);
    for (int i = 0; i < nseg; ++i) {
        const char* txt = whisper_full_get_segment_text(ctx_, i);
        if (txt) {
            if (i > 0) oss << ' ';
            oss << txt;
        }
    }

    result.text = oss.str();
    // trim leading whitespace that whisper often adds
    const auto pos = result.text.find_first_not_of(" \t");
    if (pos != std::string::npos) result.text = result.text.substr(pos);

    result.ok = true;
    return result;
}

}  // namespace pipeline
