/// @file
/// @brief Result of one transcription call, shared by the ASR adapter and its callers.
/// @ingroup asr

#pragma once

#include <functional>
#include <string>

namespace edge_audio {

/// @defgroup asr Speech recognition
/// @brief The whisper.cpp adapter and the interface the rest of the runtime uses.
///
/// Callers depend only on Transcriber (a callable), never on AsrEngine, so the
/// core library and its tests build and run without whisper.cpp or weights.
/// @{

/// @brief Outcome of transcribing one buffer.
struct AsrResult {
    std::string text;           ///< Transcript (empty on failure).
    double inference_ms = 0.0;  ///< Wall-clock decode time.
    bool ok = false;            ///< Whether decoding succeeded.
    std::string error;          ///< Failure description when `ok` is false.
};

/// @brief Transcribe mono 16 kHz float PCM. Implemented by AsrEngine in
/// production and by fakes in tests.
using Transcriber = std::function<AsrResult(const float* samples, int sample_count)>;

/// @}

}  // namespace edge_audio
