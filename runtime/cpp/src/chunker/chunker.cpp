// chunker.cpp
// Splits an AudioBuffer into fixed-size, optionally overlapping Chunks.

#include "chunker.h"

#include <algorithm>
#include <cmath>

namespace pipeline {

std::vector<Chunk> chunk_audio(const AudioBuffer& buf, const ChunkerConfig& cfg) {
    std::vector<Chunk> chunks;

    if (buf.samples.empty() || buf.sample_rate == 0) {
        return chunks;
    }

    const int sr         = buf.sample_rate;
    const int chunk_samp = static_cast<int>(std::round(cfg.chunk_ms * sr / 1000.0));
    const int hop_ms     = cfg.hop_ms > 0 ? cfg.hop_ms : cfg.chunk_ms;
    const int hop_samp   = static_cast<int>(std::round(hop_ms * sr / 1000.0));

    if (chunk_samp <= 0 || hop_samp <= 0) {
        return chunks;
    }

    const int total = static_cast<int>(buf.samples.size());
    int idx         = 0;
    int chunk_index = 0;

    while (idx < total) {
        const int end    = std::min(idx + chunk_samp, total);
        const int length = end - idx;

        if (length <= 0) break;

        Chunk c;
        c.sample_rate = sr;
        c.index       = chunk_index++;
        c.start_sec   = static_cast<double>(idx) / sr;
        c.end_sec     = static_cast<double>(end) / sr;
        c.samples.assign(buf.samples.begin() + idx, buf.samples.begin() + end);

        chunks.push_back(std::move(c));
        idx += hop_samp;
    }

    return chunks;
}

}  // namespace pipeline
