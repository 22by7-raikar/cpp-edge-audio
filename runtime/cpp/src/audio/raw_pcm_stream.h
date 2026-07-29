#pragma once

#include <array>
#include <cstdint>
#include <istream>
#include <string>
#include <vector>

namespace pipeline {

// Decodes signed 16-bit little-endian (S16LE) PCM from a std::istream
// incrementally, producing bounded blocks of normalized mono float32 samples
// in [-1.0, 1.0].
//
// Ownership: the caller owns the istream and must ensure it outlives this
// object.  RawPcmDecoder holds a non-owning reference; it never closes the
// stream.
//
// Buffer bound: the internal hold buffer is std::array<uint8_t, kMaxHoldBytes>
// (3 bytes).  It accumulates partial frame bytes between in_.get() calls and
// is reset to size zero after every complete frame.  Its size is a
// compile-time constant independent of stream duration.
//
// Does not perform VAD, utterance segmentation, DSP analysis, quality
// inference, or ASR.  Does not resample; sample_rate is retained as metadata.
class RawPcmDecoder {
public:
    // Upper bound accepted for sample_rate validation.
    static constexpr int kMaxSampleRate  = 384'000;
    // Default output block size in mono frames.
    static constexpr int kDefaultBlockFrames = 512;
    // Size of the partial-frame hold buffer (stereo worst case: 4-1 = 3 bytes).
    static constexpr int kMaxHoldBytes   = 3;

    struct Config {
        int sample_rate;                          // Input Hz; metadata only, no resampling.
        int channels;                             // 1 = mono, 2 = stereo.
        int block_frames = kDefaultBlockFrames;   // Max mono output frames per read_block.
    };

    struct Counters {
        uint64_t bytes_read           = 0;  // Raw bytes consumed from the istream.
        uint64_t input_samples        = 0;  // Individual int16 samples decoded.
        uint64_t channel_frames       = 0;  // Complete interleaved L[/R] frames decoded.
        uint64_t mono_samples_emitted = 0;  // Float samples written to callers.
        uint64_t malformed_bytes      = 0;  // Trailing bytes detected at malformed EOF.
    };

    // Construct and validate config.
    // Throws std::invalid_argument if sample_rate, channels, or block_frames
    // are out of range.  Does not read from the stream.
    explicit RawPcmDecoder(std::istream& in, Config cfg);

    // Read up to cfg.block_frames normalized mono float samples into out.
    //
    // Returns true when at least one sample is available; out is non-empty.
    // Returns false at clean EOF (no incomplete frame); out is empty, err empty.
    // Returns false on stream error; err is set, out is empty.
    //
    // Malformed-EOF handling:
    //   If a valid block of frames was decoded before hitting the malformed
    //   trailing bytes, those frames are emitted (return true, err empty) and
    //   the error is deferred: the very next call returns false with err set.
    //   If no complete frame was decoded before the malformed bytes, err is set
    //   immediately and the call returns false.
    //   malformed_bytes is updated as soon as the trailing bytes are observed.
    //
    // Once false is returned the decoder is permanently done; subsequent calls
    // return false immediately.
    bool read_block(std::vector<float>& out, std::string& err);

    int             sample_rate() const noexcept { return cfg_.sample_rate; }
    int             channels()    const noexcept { return cfg_.channels;    }
    const Counters& counters()    const noexcept { return counters_;        }

private:
    std::istream& in_;
    Config        cfg_;
    Counters      counters_{};

    // Partial-frame accumulation buffer.
    // Mono:   frame_bytes = 2, hold needs at most 1 byte.
    // Stereo: frame_bytes = 4, hold needs at most 3 bytes.
    std::array<uint8_t, kMaxHoldBytes> hold_{};
    int  hold_size_  = 0;
    bool done_       = false;
    std::string pending_err_;  // deferred malformed-EOF message
};

}  // namespace pipeline
