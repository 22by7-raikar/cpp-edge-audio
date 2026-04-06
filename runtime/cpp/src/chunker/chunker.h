#pragma once

#include <vector>
#include "audio/audio_io.h"

namespace pipeline {

// A single fixed-size segment of audio with provenance metadata.
struct Chunk {
    std::vector<float> samples;   // mono float32 PCM
    int sample_rate = 0;
    int index       = 0;          // chunk index in the stream, 0-based
    double start_sec = 0.0;       // start time relative to the source audio
    double end_sec   = 0.0;

    double duration_sec() const { return end_sec - start_sec; }
};

struct ChunkerConfig {
    int chunk_ms = 5000;  // chunk length in milliseconds
    int hop_ms   = 0;     // hop between chunks, 0 = no overlap (hop == chunk_ms)
    // TODO(M2): add min_chunk_ms threshold to drop trailing fragments below a size
};

// Split an AudioBuffer into fixed-size Chunks.
// The final chunk may be shorter than chunk_ms if the audio does not divide evenly.
// Chunks with zero samples are not emitted.
std::vector<Chunk> chunk_audio(const AudioBuffer& buf, const ChunkerConfig& cfg);

}  // namespace pipeline
