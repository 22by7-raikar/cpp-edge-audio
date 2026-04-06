#pragma once

#include <string>
#include <vector>

// Forward-declare to avoid pulling whisper.h into every translation unit.
struct whisper_context;

namespace pipeline {

struct AsrConfig {
    std::string model_path;    // path to .bin model file
    int         n_threads = 4;
    bool        translate = false;  // false = transcribe in source language
    std::string language  = "en";   // hint; whisper auto-detects if set to "auto"
    bool        no_timestamps = true;  // suppress per-token timestamps for speed
};

struct AsrResult {
    std::string text;
    double      inference_ms = 0.0;   // wall-clock time for whisper_full()
    bool        ok           = false;
    std::string error;
};

// Lifecycle wrapper for a whisper_context.
// One AsrEngine per model; reuse across chunks.
class AsrEngine {
public:
    explicit AsrEngine(const AsrConfig& cfg);
    ~AsrEngine();

    // Non-copyable
    AsrEngine(const AsrEngine&)            = delete; //copy constructor
    AsrEngine& operator=(const AsrEngine&) = delete; //copy assignment operator

    // Returns true if the model loaded successfully.
    bool ready() const { return ctx_ != nullptr; }
    const std::string& load_error() const { return load_error_; }

    // Run transcription on mono float32 PCM at 16kHz.
    // Whisper requires 16 kHz input; caller must ensure this or results are wrong.
    AsrResult transcribe(const float* samples, int n_samples);

private:
    whisper_context* ctx_       = nullptr;
    AsrConfig        cfg_;
    std::string      load_error_;
};

}  // namespace pipeline
