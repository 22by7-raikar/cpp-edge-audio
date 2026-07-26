// raw_pcm_stream.cpp
// Incremental S16LE PCM decoder.  See raw_pcm_stream.h for the full contract.

#include "raw_pcm_stream.h"

#include <cassert>
#include <ios>
#include <stdexcept>
#include <string>

namespace pipeline {

// Normalization: identical convention to audio_io.cpp / dr_wav.
// int16 / 32768.0f maps [-32768 .. 32767] -> [-1.0 .. ~1.0).
static constexpr float kNormScale = 1.0f / 32768.0f;

// Stereo downmix scale: kNormScale * 0.5 = 1 / 65536.
// Applied to the int32 sum of two int16 channels to avoid overflow.
static constexpr float kStereoScale = kNormScale * 0.5f;

static_assert(RawPcmDecoder::kMaxHoldBytes >= 2 * 2 - 1,
              "kMaxHoldBytes too small for stereo frame remainder (need 3)");

// Decode a little-endian int16 from two consecutive bytes.
static inline int16_t decode_le16(const uint8_t* p) noexcept {
    return static_cast<int16_t>(
        static_cast<uint16_t>(p[0]) |
        (static_cast<uint16_t>(p[1]) << 8));
}

RawPcmDecoder::RawPcmDecoder(std::istream& in, Config cfg)
    : in_(in), cfg_(cfg)
{
    if (cfg_.sample_rate <= 0 || cfg_.sample_rate > kMaxSampleRate) {
        throw std::invalid_argument(
            "RawPcmDecoder: sample_rate " + std::to_string(cfg_.sample_rate) +
            " out of range [1, " + std::to_string(kMaxSampleRate) + "]");
    }
    if (cfg_.channels != 1 && cfg_.channels != 2) {
        throw std::invalid_argument(
            "RawPcmDecoder: channels must be 1 or 2, got " +
            std::to_string(cfg_.channels));
    }
    if (cfg_.block_frames <= 0) {
        throw std::invalid_argument(
            "RawPcmDecoder: block_frames must be > 0, got " +
            std::to_string(cfg_.block_frames));
    }
    // Verify the hold buffer is large enough for the configured channel count.
    assert(cfg_.channels * 2 - 1 <= kMaxHoldBytes);
}

bool RawPcmDecoder::read_block(std::vector<float>& out, std::string& err) {
    out.clear();
    err.clear();

    if (done_) {
        // Deliver a deferred malformed-EOF error if one is pending.
        if (!pending_err_.empty()) {
            err = std::move(pending_err_);
        }
        return false;
    }

    const int frame_bytes = cfg_.channels * 2;
    out.reserve(static_cast<size_t>(cfg_.block_frames));

    while (static_cast<int>(out.size()) < cfg_.block_frames) {
        // Accumulate bytes into hold_ one at a time until a complete frame is
        // available.  in_.get() is used so that any streambuf chunk size,
        // including 1 byte per underflow(), works without special handling.
        while (hold_size_ < frame_bytes) {
            const int c = in_.get();

            if (c == std::char_traits<char>::eof()) {
                if (in_.bad()) {
                    // Unrecoverable I/O error; discard any partial block.
                    err  = "RawPcmDecoder: stream I/O error";
                    done_ = true;
                    out.clear();
                    return false;
                }

                // EOF reached.  Trailing bytes in hold_ are malformed data.
                if (hold_size_ > 0) {
                    counters_.malformed_bytes +=
                        static_cast<uint64_t>(hold_size_);
                    pending_err_ =
                        "RawPcmDecoder: malformed EOF — " +
                        std::to_string(hold_size_) + " trailing byte(s)";
                    hold_size_ = 0;
                    done_      = true;

                    if (!out.empty()) {
                        // Emit the frames decoded so far; caller gets the error
                        // on the next read_block() call.
                        counters_.mono_samples_emitted +=
                            static_cast<uint64_t>(out.size());
                        return true;
                    }
                    // No complete frames yet: report immediately.
                    err = std::move(pending_err_);
                    return false;
                }

                // Clean EOF with no partial frame.
                done_ = true;
                if (!out.empty()) {
                    counters_.mono_samples_emitted +=
                        static_cast<uint64_t>(out.size());
                    return true;
                }
                return false;
            }

            hold_[hold_size_++] = static_cast<uint8_t>(c);
            ++counters_.bytes_read;
        }

        // Decode one complete frame.
        if (cfg_.channels == 1) {
            const int16_t s = decode_le16(hold_.data());
            out.push_back(static_cast<float>(s) * kNormScale);
            ++counters_.input_samples;
        } else {
            // Stereo: average L and R via int32 to avoid int16 overflow.
            const int16_t l = decode_le16(hold_.data());
            const int16_t r = decode_le16(hold_.data() + 2);
            const int32_t mix =
                static_cast<int32_t>(l) + static_cast<int32_t>(r);
            out.push_back(static_cast<float>(mix) * kStereoScale);
            counters_.input_samples += 2;
        }
        ++counters_.channel_frames;
        hold_size_ = 0;
    }

    counters_.mono_samples_emitted += static_cast<uint64_t>(out.size());
    return true;
}

}  // namespace pipeline
