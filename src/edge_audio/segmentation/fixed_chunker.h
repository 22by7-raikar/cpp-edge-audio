/// @file
/// @brief Fixed-length windowing of a complete buffer.
/// @ingroup segmentation

#pragma once

#include <vector>

#include "edge_audio/audio/audio_io.h"

namespace edge_audio {

/// @defgroup segmentation Segmentation
/// @brief Splitting audio into the units that are analysed and transcribed.
///
/// - chunk_audio(): fixed windows over a complete file (offline pipeline; the
///   learned quality model is trained on non-overlapping 5 s windows).
/// - run_vad() / pack_vad_segments(): energy/ZCR speech regions over a complete
///   file, optionally packed into fewer, wider ASR windows.
/// - StreamingUtteranceSegmenter: bounded-memory endpointing of a live stream
///   into utterances (streaming pipeline).
/// @{

/// @brief A window of mono audio and its position in the source.
struct Chunk {
    std::vector<float> samples;  ///< Mono PCM.
    int sample_rate = 0;         ///< Hz.
    int index = 0;               ///< 0-based position in the chunk sequence.
    double start_sec = 0.0;      ///< Start time in the source, seconds.
    double end_sec = 0.0;        ///< End time in the source, seconds (exclusive).

    /// @brief Window length in seconds.
    double duration_sec() const { return end_sec - start_sec; }
};

/// @brief Window length and hop for chunk_audio().
struct ChunkerConfig {
    int chunk_ms = 5000;  ///< Window length in milliseconds.
    int hop_ms = 0;       ///< Hop in milliseconds; 0 means no overlap (hop = chunk_ms).
};

/// @brief Split @p buf into consecutive windows.
///
/// The last window is shorter when the audio does not divide evenly; empty
/// windows are never produced.
std::vector<Chunk> chunk_audio(const AudioBuffer& buf, const ChunkerConfig& cfg);

/// @}

}  // namespace edge_audio
