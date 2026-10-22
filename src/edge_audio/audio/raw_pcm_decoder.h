/// @file
/// @brief Incremental decoder for raw S16LE PCM streams.
/// @ingroup audio

#pragma once

#include <array>
#include <cstdint>
#include <istream>
#include <string>
#include <vector>

namespace edge_audio {

/// @addtogroup audio
/// @{

/// @brief Decodes interleaved signed 16-bit little-endian PCM from a stream
/// into bounded blocks of mono float samples.
///
/// Memory use is constant regardless of stream length: a partial frame that
/// straddles two reads is held in a 3-byte buffer, and each call emits at most
/// Config::block_frames samples. Stereo input is downmixed by averaging.
/// The decoder performs no resampling; `sample_rate` is carried as metadata.
///
/// The decoder holds a non-owning reference to the input stream, which must
/// outlive it.
///
/// @par Error model
/// Once read_block() returns `false` the decoder is finished and every later
/// call returns `false`. Trailing bytes that do not form a complete frame are a
/// malformed stream: frames decoded before them are still delivered, and the
/// error is reported on the following call.
class RawPcmDecoder {
public:
    static constexpr int kMaxSampleRate = 384'000;   ///< Largest accepted sample rate.
    static constexpr int kDefaultBlockFrames = 512;  ///< Default output block size.
    static constexpr int kMaxHoldBytes = 3;          ///< Partial-frame buffer (stereo frame - 1).

    /// @brief Stream format.
    struct Config {
        int sample_rate;                         ///< Input rate in Hz, in [1, #kMaxSampleRate].
        int channels;                            ///< 1 (mono) or 2 (interleaved stereo).
        int block_frames = kDefaultBlockFrames;  ///< Maximum frames per read_block() call.
    };

    /// @brief Running byte and sample counters, for telemetry.
    struct Counters {
        uint64_t bytes_read = 0;            ///< Bytes consumed from the stream.
        uint64_t input_samples = 0;         ///< int16 samples decoded (all channels).
        uint64_t channel_frames = 0;        ///< Complete interleaved frames decoded.
        uint64_t mono_samples_emitted = 0;  ///< Float samples returned to callers.
        uint64_t malformed_bytes = 0;       ///< Trailing bytes found at a malformed EOF.
    };

    /// @brief Validate @p cfg and bind to @p in. Does not read from the stream.
    /// @throws std::invalid_argument if any Config field is out of range.
    RawPcmDecoder(std::istream& in, Config cfg);

    /// @brief Decode up to Config::block_frames mono samples.
    ///
    /// @param out Replaced with the decoded samples (empty when returning `false`).
    /// @param err Set to a message when the stream is malformed or unreadable.
    /// @return `true` if @p out holds at least one sample; `false` at clean EOF
    ///         (@p err empty) or on error (@p err set).
    bool read_block(std::vector<float>& out, std::string& err);

    int sample_rate() const noexcept { return cfg_.sample_rate; }    ///< Configured input rate.
    int channels() const noexcept { return cfg_.channels; }          ///< Configured channel count.
    const Counters& counters() const noexcept { return counters_; }  ///< Running counters.

private:
    std::istream& in_;
    Config cfg_;
    Counters counters_{};
    std::array<uint8_t, kMaxHoldBytes> hold_{};  // bytes of an incomplete frame
    int hold_size_ = 0;
    bool done_ = false;
    std::string pending_err_;  // malformed-EOF error deferred to the next call
};

/// @}

}  // namespace edge_audio
