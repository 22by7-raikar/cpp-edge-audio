// RawPcmDecoder: normalisation, stereo downmix, arbitrary read boundaries, malformed EOF.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <istream>
#include <limits>
#include <streambuf>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "edge_audio/audio/raw_pcm_decoder.h"

using namespace edge_audio;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Pack int16 values as S16LE bytes.
static std::vector<uint8_t> s16le(std::initializer_list<int16_t> vals) {
    std::vector<uint8_t> out;
    out.reserve(vals.size() * 2);
    for (int16_t v : vals) {
        out.push_back(static_cast<uint8_t>(v & 0xFF));
        out.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
    }
    return out;
}

// streambuf backed by a fixed byte vector.
// chunk_size controls how many bytes are loaded into the get area per
// underflow() call, simulating a source that delivers data in bounded pieces.
// Because RawPcmDecoder uses in_.get() (one byte per call), all chunk sizes
// produce identical decoded output; the varying sizes exercise the streambuf /
// istream interaction paths.
struct ChunkedBuf : std::streambuf {
    std::vector<uint8_t> data_;
    size_t pos_ = 0;
    size_t chunk_;

    ChunkedBuf(std::vector<uint8_t> d, size_t chunk) : data_(std::move(d)), chunk_(chunk) {
        setg(nullptr, nullptr, nullptr);
    }

    int underflow() override {
        if (pos_ >= data_.size()) return traits_type::eof();
        size_t n = std::min(chunk_, data_.size() - pos_);
        char* base = reinterpret_cast<char*>(data_.data() + pos_);
        setg(base, base, base + n);
        pos_ += n;
        return traits_type::to_int_type(*gptr());
    }
};

// Drain all output into a flat vector; returns true on clean EOF.
static bool decode_all(RawPcmDecoder& dec, std::vector<float>& all_out, std::string& last_err) {
    std::vector<float> blk;
    std::string err;
    while (dec.read_block(blk, err)) {
        all_out.insert(all_out.end(), blk.begin(), blk.end());
        if (!err.empty()) {
            last_err = err;
            return false;
        }
    }
    last_err = err;
    return err.empty();
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST(RawPcmDecoder, EmptyCleanEof) {
    ChunkedBuf buf({}, 4096);
    std::istream in(&buf);
    RawPcmDecoder dec(in, {16000, 1});
    std::vector<float> out;
    std::string err;

    ASSERT_FALSE(dec.read_block(out, err));
    ASSERT_TRUE(out.empty());
    ASSERT_TRUE(err.empty());
    ASSERT_EQ(dec.counters().bytes_read, 0);
    ASSERT_EQ(dec.counters().input_samples, 0);
    ASSERT_EQ(dec.counters().channel_frames, 0);
    ASSERT_EQ(dec.counters().mono_samples_emitted, 0);
    ASSERT_EQ(dec.counters().malformed_bytes, 0);
}

TEST(RawPcmDecoder, MonoMinMaxZeroRepresentative) {
    // INT16_MIN, INT16_MAX, 0, 100 — all in one block.
    auto data = s16le({-32768, 32767, 0, 100});
    ChunkedBuf buf(data, 4096);
    std::istream in(&buf);
    RawPcmDecoder dec(in, {16000, 1, 16});
    std::vector<float> out;
    std::string err;

    ASSERT_TRUE(dec.read_block(out, err));
    ASSERT_TRUE(err.empty());
    ASSERT_EQ(out.size(), 4);
    ASSERT_NEAR(out[0], -1.0f, 1e-6f);                // INT16_MIN / 32768
    ASSERT_NEAR(out[1], 32767.0f / 32768.0f, 1e-6f);  // INT16_MAX
    ASSERT_NEAR(out[2], 0.0f, 1e-6f);                 // zero
    ASSERT_NEAR(out[3], 100.0f / 32768.0f, 1e-6f);    // representative
}

TEST(RawPcmDecoder, StereoNormalDecoding) {
    // L=1000, R=2000 -> downmix = (1000+2000)/65536
    auto data = s16le({1000, 2000});
    ChunkedBuf buf(data, 4096);
    std::istream in(&buf);
    RawPcmDecoder dec(in, {16000, 2, 8});
    std::vector<float> out;
    std::string err;

    ASSERT_TRUE(dec.read_block(out, err));
    ASSERT_TRUE(err.empty());
    ASSERT_EQ(out.size(), 1);

    const float expected = (1000.0f + 2000.0f) / 65536.0f;
    ASSERT_NEAR(out[0], expected, 1e-6f);

    ASSERT_EQ(dec.counters().bytes_read, 4);
    ASSERT_EQ(dec.counters().input_samples, 2);
    ASSERT_EQ(dec.counters().channel_frames, 1);
    ASSERT_EQ(dec.counters().mono_samples_emitted, 1);
}

TEST(RawPcmDecoder, StereoDownmixNoOverflow) {
    // Both channels at INT16_MAX: sum = 65534 (requires int32 intermediate).
    {
        auto data = s16le({32767, 32767});
        ChunkedBuf buf(data, 4096);
        std::istream in(&buf);
        RawPcmDecoder dec(in, {16000, 2, 4});
        std::vector<float> out;
        std::string err;

        ASSERT_TRUE(dec.read_block(out, err));
        ASSERT_EQ(out.size(), 1);
        const float expected = (32767.0f + 32767.0f) / 65536.0f;
        ASSERT_NEAR(out[0], expected, 1e-5f);
        ASSERT_LE(out[0], 1.0f);  // must not saturate above 1.0
    }
    // Both channels at INT16_MIN: sum = -65536, result = -1.0.
    {
        auto data = s16le({-32768, -32768});
        ChunkedBuf buf(data, 4096);
        std::istream in(&buf);
        RawPcmDecoder dec(in, {16000, 2, 4});
        std::vector<float> out;
        std::string err;

        ASSERT_TRUE(dec.read_block(out, err));
        ASSERT_EQ(out.size(), 1);
        ASSERT_NEAR(out[0], -1.0f, 1e-6f);
        ASSERT_GE(out[0], -1.0f);  // must not go below -1.0
    }
    // Opposite extremes: L=INT16_MAX, R=INT16_MIN -> near zero.
    {
        auto data = s16le({32767, -32768});
        ChunkedBuf buf(data, 4096);
        std::istream in(&buf);
        RawPcmDecoder dec(in, {16000, 2, 4});
        std::vector<float> out;
        std::string err;

        ASSERT_TRUE(dec.read_block(out, err));
        ASSERT_EQ(out.size(), 1);
        const float expected = (32767.0f + (-32768.0f)) / 65536.0f;
        ASSERT_NEAR(out[0], expected, 1e-6f);
    }
}

TEST(RawPcmDecoder, MultipleOutputBlocks) {
    // 1024 mono samples, block_frames=512 -> 2 full blocks, then clean EOF.
    std::vector<uint8_t> data;
    data.reserve(1024 * 2);
    for (int i = 0; i < 1024; ++i) {
        const int16_t v = static_cast<int16_t>(i & 0x7FFF);
        data.push_back(static_cast<uint8_t>(v & 0xFF));
        data.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
    }
    ChunkedBuf buf(data, 4096);
    std::istream in(&buf);
    RawPcmDecoder dec(in, {16000, 1, 512});

    std::vector<float> out;
    std::string err;

    ASSERT_TRUE(dec.read_block(out, err));
    ASSERT_EQ(out.size(), 512);
    ASSERT_TRUE(err.empty());

    ASSERT_TRUE(dec.read_block(out, err));
    ASSERT_EQ(out.size(), 512);
    ASSERT_TRUE(err.empty());

    ASSERT_FALSE(dec.read_block(out, err));
    ASSERT_TRUE(out.empty());
    ASSERT_TRUE(err.empty());

    ASSERT_EQ(dec.counters().channel_frames, 1024);
    ASSERT_EQ(dec.counters().mono_samples_emitted, 1024);
}

TEST(RawPcmDecoder, OneByteUnderlyingReads) {
    // chunk=1: each underflow() delivers 1 byte.  Decoder must assemble frames
    // correctly from single-byte pieces.
    auto data = s16le({-100, 200, -300});
    ChunkedBuf buf(data, 1);
    std::istream in(&buf);
    RawPcmDecoder dec(in, {16000, 1, 8});
    std::vector<float> out;
    std::string err;

    ASSERT_TRUE(dec.read_block(out, err));
    ASSERT_EQ(out.size(), 3);
    ASSERT_TRUE(err.empty());
    ASSERT_NEAR(out[0], -100.0f / 32768.0f, 1e-6f);
    ASSERT_NEAR(out[1], 200.0f / 32768.0f, 1e-6f);
    ASSERT_NEAR(out[2], -300.0f / 32768.0f, 1e-6f);
}

TEST(RawPcmDecoder, ThreeByteUnderlyingReads) {
    // chunk=3: non-aligned to the 2-byte frame boundary.
    auto data = s16le({1000, 2000, 3000, 4000});
    ChunkedBuf buf(data, 3);
    std::istream in(&buf);
    RawPcmDecoder dec(in, {16000, 1, 8});
    std::vector<float> out;
    std::string err;

    ASSERT_TRUE(dec.read_block(out, err));
    ASSERT_EQ(out.size(), 4);
    ASSERT_NEAR(out[0], 1000.0f / 32768.0f, 1e-6f);
    ASSERT_NEAR(out[3], 4000.0f / 32768.0f, 1e-6f);
}

TEST(RawPcmDecoder, FiveByteUnderlyingReads) {
    // chunk=5: non-aligned to both mono (2-byte) and stereo (4-byte) frames.
    auto data = s16le({-1, -2, -3, -4, -5});
    ChunkedBuf buf(data, 5);
    std::istream in(&buf);
    RawPcmDecoder dec(in, {16000, 1, 8});
    std::vector<float> out;
    std::string err;

    ASSERT_TRUE(dec.read_block(out, err));
    ASSERT_EQ(out.size(), 5);
    for (int i = 0; i < 5; ++i) {
        ASSERT_NEAR(out[i], static_cast<float>(-(i + 1)) / 32768.0f, 1e-6f);
    }
}

TEST(RawPcmDecoder, Int16SplitAcrossReads) {
    // Mono: 1 sample = 2 bytes.  chunk=1 forces each byte to arrive via a
    // separate underflow(); hold_ must accumulate both before decoding.
    const int16_t val = 1234;
    const std::vector<uint8_t> data = {static_cast<uint8_t>(val & 0xFF),
                                       static_cast<uint8_t>((val >> 8) & 0xFF)};
    ChunkedBuf buf(data, 1);
    std::istream in(&buf);
    RawPcmDecoder dec(in, {16000, 1, 4});
    std::vector<float> out;
    std::string err;

    ASSERT_TRUE(dec.read_block(out, err));
    ASSERT_EQ(out.size(), 1);
    ASSERT_NEAR(out[0], 1234.0f / 32768.0f, 1e-6f);
}

TEST(RawPcmDecoder, StereoFrameSplitAcrossReads) {
    // Stereo: 1 frame = 4 bytes.  chunk=2 splits each frame across 2
    // underflow() calls (L bytes on first, R bytes on second).
    const int16_t L = 500, R = 700;
    const std::vector<uint8_t> data = {
        static_cast<uint8_t>(L & 0xFF), static_cast<uint8_t>((L >> 8) & 0xFF),
        static_cast<uint8_t>(R & 0xFF), static_cast<uint8_t>((R >> 8) & 0xFF)};
    ChunkedBuf buf(data, 2);
    std::istream in(&buf);
    RawPcmDecoder dec(in, {16000, 2, 4});
    std::vector<float> out;
    std::string err;

    ASSERT_TRUE(dec.read_block(out, err));
    ASSERT_EQ(out.size(), 1);
    const float expected = (500.0f + 700.0f) / 65536.0f;
    ASSERT_NEAR(out[0], expected, 1e-6f);
}

TEST(RawPcmDecoder, OddTrailingByteAtEof) {
    // 1 valid mono sample (2 bytes) followed by 1 orphan byte.
    // Expected: first call emits the valid sample (true); second call reports
    // the malformed-EOF error (false).
    auto data = s16le({42});
    data.push_back(0xFF);  // orphan
    ChunkedBuf buf(data, 4096);
    std::istream in(&buf);
    RawPcmDecoder dec(in, {16000, 1, 8});
    std::vector<float> out;
    std::string err;

    // First call: valid sample emitted; error deferred.
    const bool r1 = dec.read_block(out, err);
    ASSERT_EQ(r1, true);
    ASSERT_EQ(out.size(), 1);
    ASSERT_TRUE(err.empty());
    ASSERT_NEAR(out[0], 42.0f / 32768.0f, 1e-6f);
    ASSERT_EQ(dec.counters().malformed_bytes, 1);  // already counted

    // Second call: deferred error is delivered.
    const bool r2 = dec.read_block(out, err);
    ASSERT_EQ(r2, false);
    ASSERT_TRUE(out.empty());
    ASSERT_FALSE(err.empty());
    ASSERT_EQ(dec.counters().malformed_bytes, 1);  // unchanged
}

TEST(RawPcmDecoder, IncompleteStereoFrameAtEof) {
    // 2 bytes: half a stereo frame (needs 4).  No complete frame decoded,
    // so the error is immediate.
    const std::vector<uint8_t> data = {0x01, 0x00};
    ChunkedBuf buf(data, 4096);
    std::istream in(&buf);
    RawPcmDecoder dec(in, {16000, 2, 4});
    std::vector<float> out;
    std::string err;

    const bool r = dec.read_block(out, err);
    ASSERT_EQ(r, false);
    ASSERT_TRUE(out.empty());
    ASSERT_FALSE(err.empty());
    ASSERT_EQ(dec.counters().malformed_bytes, 2);
    ASSERT_EQ(dec.counters().channel_frames, 0);
}

TEST(RawPcmDecoder, InvalidSampleRate) {
    ChunkedBuf buf({}, 1);
    std::istream in(&buf);
    bool threw = false;

    // Zero
    threw = false;
    try {
        RawPcmDecoder dec(in, {0, 1});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    ASSERT_TRUE(threw);

    // Negative
    threw = false;
    try {
        RawPcmDecoder dec(in, {-44100, 1});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    ASSERT_TRUE(threw);

    // Unreasonably high (above kMaxSampleRate)
    threw = false;
    try {
        RawPcmDecoder dec(in, {RawPcmDecoder::kMaxSampleRate + 1, 1});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    ASSERT_TRUE(threw);

    // Valid boundary: minimum (1 Hz)
    threw = false;
    try {
        RawPcmDecoder dec(in, {1, 1});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    ASSERT_FALSE(threw);

    // Valid boundary: maximum
    threw = false;
    try {
        RawPcmDecoder dec(in, {RawPcmDecoder::kMaxSampleRate, 1});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    ASSERT_FALSE(threw);
}

TEST(RawPcmDecoder, InvalidChannelCount) {
    ChunkedBuf buf({}, 1);
    std::istream in(&buf);
    bool threw = false;

    threw = false;
    try {
        RawPcmDecoder dec(in, {16000, 0});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    ASSERT_TRUE(threw);

    threw = false;
    try {
        RawPcmDecoder dec(in, {16000, 3});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    ASSERT_TRUE(threw);

    threw = false;
    try {
        RawPcmDecoder dec(in, {16000, -1});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    ASSERT_TRUE(threw);

    // Valid: 1 and 2
    threw = false;
    try {
        RawPcmDecoder dec(in, {16000, 1});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    ASSERT_FALSE(threw);

    threw = false;
    try {
        RawPcmDecoder dec(in, {16000, 2});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    ASSERT_FALSE(threw);
}

TEST(RawPcmDecoder, CounterAccuracyMono) {
    // 6 mono samples, block_frames=3 -> 2 blocks then clean EOF.
    auto data = s16le({10, 20, 30, 40, 50, 60});
    ChunkedBuf buf(data, 4096);
    std::istream in(&buf);
    RawPcmDecoder dec(in, {16000, 1, 3});

    std::vector<float> out;
    std::string err;

    ASSERT_TRUE(dec.read_block(out, err));
    ASSERT_EQ(out.size(), 3);
    ASSERT_TRUE(dec.read_block(out, err));
    ASSERT_EQ(out.size(), 3);
    ASSERT_FALSE(dec.read_block(out, err));
    ASSERT_TRUE(err.empty());

    const auto& c = dec.counters();
    ASSERT_EQ(c.bytes_read, 12);
    ASSERT_EQ(c.input_samples, 6);
    ASSERT_EQ(c.channel_frames, 6);
    ASSERT_EQ(c.mono_samples_emitted, 6);
    ASSERT_EQ(c.malformed_bytes, 0);
}

TEST(RawPcmDecoder, CounterAccuracyStereo) {
    // 3 stereo frames.  input_samples = 6, channel_frames = 3, emitted = 3.
    auto data = s16le({100, -100, 200, -200, 300, -300});
    ChunkedBuf buf(data, 4096);
    std::istream in(&buf);
    RawPcmDecoder dec(in, {44100, 2, 8});

    std::vector<float> all;
    std::string err;
    ASSERT_TRUE(decode_all(dec, all, err));
    ASSERT_EQ(all.size(), 3);

    const auto& c = dec.counters();
    ASSERT_EQ(c.bytes_read, 12);    // 3 * 2ch * 2 bytes
    ASSERT_EQ(c.input_samples, 6);  // 3 * 2ch
    ASSERT_EQ(c.channel_frames, 3);
    ASSERT_EQ(c.mono_samples_emitted, 3);
    ASSERT_EQ(c.malformed_bytes, 0);
}

TEST(RawPcmDecoder, InternalBufferBounded) {
    // Decode a large stereo stream with a non-aligned chunk size to stress the
    // hold_ accumulation logic.  Verify that counters are fully consistent:
    // because hold_ is always reset to zero after each complete frame, any
    // bytes remaining at clean EOF would appear as malformed_bytes — so
    // malformed_bytes == 0 proves hold_ was correctly drained.
    //
    // hold_ is std::array<uint8_t, RawPcmDecoder::kMaxHoldBytes> — 3 bytes.
    // Its size is a compile-time constant; it does not grow with stream length.
    static_assert(RawPcmDecoder::kMaxHoldBytes == 3, "bound proof relies on kMaxHoldBytes == 3");

    constexpr int N = 5000;  // stereo frames
    std::vector<uint8_t> data;
    data.reserve(N * 4);
    for (int i = 0; i < N; ++i) {
        const int16_t L = static_cast<int16_t>(i & 0x7FFF);
        const int16_t R = static_cast<int16_t>((i * 7) & 0x7FFF);
        data.push_back(static_cast<uint8_t>(L & 0xFF));
        data.push_back(static_cast<uint8_t>((L >> 8) & 0xFF));
        data.push_back(static_cast<uint8_t>(R & 0xFF));
        data.push_back(static_cast<uint8_t>((R >> 8) & 0xFF));
    }
    // chunk=7: non-aligned to frame_bytes=4, exercises partial-frame hold.
    ChunkedBuf buf(data, 7);
    std::istream in(&buf);
    RawPcmDecoder dec(in, {16000, 2, 64});

    std::vector<float> all;
    std::string err;
    ASSERT_TRUE(decode_all(dec, all, err));
    ASSERT_TRUE(err.empty());
    ASSERT_EQ(all.size(), static_cast<size_t>(N));

    const auto& c = dec.counters();
    ASSERT_EQ(c.bytes_read, static_cast<uint64_t>(N * 4));
    ASSERT_EQ(c.input_samples, static_cast<uint64_t>(N * 2));
    ASSERT_EQ(c.channel_frames, static_cast<uint64_t>(N));
    ASSERT_EQ(c.mono_samples_emitted, static_cast<uint64_t>(N));
    ASSERT_EQ(c.malformed_bytes, 0);  // proves hold_ was fully drained
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
