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
    // whisper_init_from_file is available across all whisper.cpp versions.
    // TODO(M3): switch to whisper_init_from_file_with_params when
    //           GPU/CUDA/CoreML context params are needed.
    ctx_ = whisper_init_from_file(cfg.model_path.c_str());
    if (!ctx_) {
        load_error_ = "whisper_init_from_file failed: " + cfg.model_path;
    }
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
