/// @file
/// @brief RAII wrapper around a whisper.cpp context.
/// @ingroup asr

#pragma once

#include <string>

#include "edge_audio/asr/transcription.h"

struct whisper_context;  // whisper.h is kept out of public headers

namespace edge_audio {

/// @addtogroup asr
/// @{

/// @brief AsrEngine settings.
struct AsrConfig {
    std::string model_path;       ///< whisper.cpp GGML model (`ggml-*.bin`).
    int n_threads = 4;            ///< Decoder threads.
    bool translate = false;       ///< Translate to English instead of transcribing.
    std::string language = "en";  ///< Language hint, or `auto` to detect.
    bool no_timestamps = true;    ///< Skip timestamp tokens (faster).
};

/// @brief Owns one whisper.cpp context and transcribes buffers with greedy decoding.
///
/// When built with `GGML_CUDA=ON` the context is created on GPU 0 and falls
/// back to the CPU if that fails; backend_mode() reports what is in use.
/// Construct once and reuse: loading the model dominates start-up time.
/// Not thread-safe; one call at a time.
class AsrEngine {
public:
    /// @brief Load the model. Check ready() / load_error() afterwards; the
    /// constructor does not throw on a missing or invalid model.
    explicit AsrEngine(const AsrConfig& cfg);
    ~AsrEngine();

    AsrEngine(const AsrEngine&) = delete;
    AsrEngine& operator=(const AsrEngine&) = delete;

    bool ready() const { return ctx_ != nullptr; }                 ///< Model loaded.
    const std::string& load_error() const { return load_error_; }  ///< Why loading failed.
    bool gpu_requested() const { return gpu_requested_; }          ///< Built with GPU support.
    bool gpu_enabled() const { return gpu_enabled_; }              ///< Running on the GPU.
    bool used_cpu_fallback() const { return cpu_fallback_; }       ///< GPU init failed; on CPU.
    /// @brief Active backend: `cpu`, `cuda` or `unavailable`.
    const std::string& backend_mode() const { return backend_mode_; }

    /// @brief Transcribe mono float PCM at 16 kHz (whisper's only input rate).
    AsrResult transcribe(const float* samples, int n_samples);

    /// @brief This engine as a Transcriber. The engine must outlive the callable.
    Transcriber as_transcriber() {
        return [this](const float* samples, int n) { return transcribe(samples, n); };
    }

private:
    whisper_context* ctx_ = nullptr;
    AsrConfig cfg_;
    std::string load_error_;
    bool gpu_requested_ = false;
    bool gpu_enabled_ = false;
    bool cpu_fallback_ = false;
    std::string backend_mode_ = "cpu";
};

/// @}

}  // namespace edge_audio
