// test_raw_pcm_stream.cpp
// Deterministic unit tests for RawPcmDecoder.
// No external framework.  Run via CTest: ctest -R raw_pcm_stream --output-on-failure

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <istream>
#include <iostream>
#include <limits>
#include <streambuf>
#include <string>
#include <vector>

#include "audio/raw_pcm_stream.h"

using namespace pipeline;

// ---------------------------------------------------------------------------
// Minimal test harness (mirrors test_main.cpp style)
// ---------------------------------------------------------------------------

static int g_passed = 0;
static int g_failed = 0;

#define CHECK(cond)                                                        \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::cerr << "FAIL  " << __FILE__ << ":" << __LINE__          \
                      << "  " << #cond << "\n";                           \
            return false;                                                  \
        }                                                                  \
    } while (0)

#define CHECK_NEAR(a, b, tol)                                                \
    do {                                                                     \
        float _a = static_cast<float>(a);                                    \
        float _b = static_cast<float>(b);                                    \
        float _t = static_cast<float>(tol);                                  \
        if (std::fabs(_a - _b) > _t) {                                      \
            std::cerr << "FAIL  " << __FILE__ << ":" << __LINE__            \
                      << "  |" << _a << " - " << _b << "| > " << _t << "\n"; \
            return false;                                                     \
        }                                                                     \
    } while (0)

static bool run(const char* name, bool (*fn)()) {
    const bool ok = fn();
    if (ok) { ++g_passed; std::cout << "  PASS  " << name << "\n"; }
    else    { ++g_failed; std::cout << "  FAIL  " << name << "\n"; }
    return ok;
}

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
    size_t pos_   = 0;
    size_t chunk_;

    ChunkedBuf(std::vector<uint8_t> d, size_t chunk)
        : data_(std::move(d)), chunk_(chunk)
    { setg(nullptr, nullptr, nullptr); }

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
static bool decode_all(RawPcmDecoder& dec,
                       std::vector<float>& all_out,
                       std::string& last_err)
{
    std::vector<float> blk;
    std::string err;
    while (dec.read_block(blk, err)) {
        all_out.insert(all_out.end(), blk.begin(), blk.end());
        if (!err.empty()) { last_err = err; return false; }
    }
    last_err = err;
    return err.empty();
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

static bool test_empty_clean_eof() {
    ChunkedBuf buf({}, 4096);
    std::istream in(&buf);
    RawPcmDecoder dec(in, {16000, 1});
    std::vector<float> out;
    std::string err;

    CHECK(!dec.read_block(out, err));
    CHECK(out.empty());
    CHECK(err.empty());
    CHECK(dec.counters().bytes_read          == 0);
    CHECK(dec.counters().input_samples       == 0);
    CHECK(dec.counters().channel_frames      == 0);
    CHECK(dec.counters().mono_samples_emitted== 0);
    CHECK(dec.counters().malformed_bytes     == 0);
    return true;
}

static bool test_mono_min_max_zero_representative() {
    // INT16_MIN, INT16_MAX, 0, 100 — all in one block.
    auto data = s16le({-32768, 32767, 0, 100});
    ChunkedBuf buf(data, 4096);
    std::istream in(&buf);
    RawPcmDecoder dec(in, {16000, 1, 16});
    std::vector<float> out;
    std::string err;

    CHECK(dec.read_block(out, err));
    CHECK(err.empty());
    CHECK(out.size() == 4);
    CHECK_NEAR(out[0], -1.0f,                  1e-6f);   // INT16_MIN / 32768
    CHECK_NEAR(out[1], 32767.0f / 32768.0f,    1e-6f);   // INT16_MAX
    CHECK_NEAR(out[2], 0.0f,                   1e-6f);   // zero
    CHECK_NEAR(out[3], 100.0f / 32768.0f,      1e-6f);   // representative
    return true;
}

static bool test_stereo_normal_decoding() {
    // L=1000, R=2000 -> downmix = (1000+2000)/65536
    auto data = s16le({1000, 2000});
    ChunkedBuf buf(data, 4096);
    std::istream in(&buf);
    RawPcmDecoder dec(in, {16000, 2, 8});
    std::vector<float> out;
    std::string err;

    CHECK(dec.read_block(out, err));
    CHECK(err.empty());
    CHECK(out.size() == 1);

    const float expected = (1000.0f + 2000.0f) / 65536.0f;
    CHECK_NEAR(out[0], expected, 1e-6f);

    CHECK(dec.counters().bytes_read           == 4);
    CHECK(dec.counters().input_samples        == 2);
    CHECK(dec.counters().channel_frames       == 1);
    CHECK(dec.counters().mono_samples_emitted == 1);
    return true;
}

static bool test_stereo_downmix_no_overflow() {
    // Both channels at INT16_MAX: sum = 65534 (requires int32 intermediate).
    {
        auto data = s16le({32767, 32767});
        ChunkedBuf buf(data, 4096);
        std::istream in(&buf);
        RawPcmDecoder dec(in, {16000, 2, 4});
        std::vector<float> out;
        std::string err;

        CHECK(dec.read_block(out, err));
        CHECK(out.size() == 1);
        const float expected = (32767.0f + 32767.0f) / 65536.0f;
        CHECK_NEAR(out[0], expected, 1e-5f);
        CHECK(out[0] <= 1.0f);   // must not saturate above 1.0
    }
    // Both channels at INT16_MIN: sum = -65536, result = -1.0.
    {
        auto data = s16le({-32768, -32768});
        ChunkedBuf buf(data, 4096);
        std::istream in(&buf);
        RawPcmDecoder dec(in, {16000, 2, 4});
        std::vector<float> out;
        std::string err;

        CHECK(dec.read_block(out, err));
        CHECK(out.size() == 1);
        CHECK_NEAR(out[0], -1.0f, 1e-6f);
        CHECK(out[0] >= -1.0f);  // must not go below -1.0
    }
    // Opposite extremes: L=INT16_MAX, R=INT16_MIN -> near zero.
    {
        auto data = s16le({32767, -32768});
        ChunkedBuf buf(data, 4096);
        std::istream in(&buf);
        RawPcmDecoder dec(in, {16000, 2, 4});
        std::vector<float> out;
        std::string err;

        CHECK(dec.read_block(out, err));
        CHECK(out.size() == 1);
        const float expected = (32767.0f + (-32768.0f)) / 65536.0f;
        CHECK_NEAR(out[0], expected, 1e-6f);
    }
    return true;
}

static bool test_multiple_output_blocks() {
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

    CHECK(dec.read_block(out, err));
    CHECK(out.size() == 512);
    CHECK(err.empty());

    CHECK(dec.read_block(out, err));
    CHECK(out.size() == 512);
    CHECK(err.empty());

    CHECK(!dec.read_block(out, err));
    CHECK(out.empty());
    CHECK(err.empty());

    CHECK(dec.counters().channel_frames       == 1024);
    CHECK(dec.counters().mono_samples_emitted == 1024);
    return true;
}

static bool test_one_byte_underlying_reads() {
    // chunk=1: each underflow() delivers 1 byte.  Decoder must assemble frames
    // correctly from single-byte pieces.
    auto data = s16le({-100, 200, -300});
    ChunkedBuf buf(data, 1);
    std::istream in(&buf);
    RawPcmDecoder dec(in, {16000, 1, 8});
    std::vector<float> out;
    std::string err;

    CHECK(dec.read_block(out, err));
    CHECK(out.size() == 3);
    CHECK(err.empty());
    CHECK_NEAR(out[0], -100.0f / 32768.0f, 1e-6f);
    CHECK_NEAR(out[1],  200.0f / 32768.0f, 1e-6f);
    CHECK_NEAR(out[2], -300.0f / 32768.0f, 1e-6f);
    return true;
}

static bool test_three_byte_underlying_reads() {
    // chunk=3: non-aligned to the 2-byte frame boundary.
    auto data = s16le({1000, 2000, 3000, 4000});
    ChunkedBuf buf(data, 3);
    std::istream in(&buf);
    RawPcmDecoder dec(in, {16000, 1, 8});
    std::vector<float> out;
    std::string err;

    CHECK(dec.read_block(out, err));
    CHECK(out.size() == 4);
    CHECK_NEAR(out[0], 1000.0f / 32768.0f, 1e-6f);
    CHECK_NEAR(out[3], 4000.0f / 32768.0f, 1e-6f);
    return true;
}

static bool test_five_byte_underlying_reads() {
    // chunk=5: non-aligned to both mono (2-byte) and stereo (4-byte) frames.
    auto data = s16le({-1, -2, -3, -4, -5});
    ChunkedBuf buf(data, 5);
    std::istream in(&buf);
    RawPcmDecoder dec(in, {16000, 1, 8});
    std::vector<float> out;
    std::string err;

    CHECK(dec.read_block(out, err));
    CHECK(out.size() == 5);
    for (int i = 0; i < 5; ++i) {
        CHECK_NEAR(out[i], static_cast<float>(-(i + 1)) / 32768.0f, 1e-6f);
    }
    return true;
}

static bool test_int16_split_across_reads() {
    // Mono: 1 sample = 2 bytes.  chunk=1 forces each byte to arrive via a
    // separate underflow(); hold_ must accumulate both before decoding.
    const int16_t val = 1234;
    const std::vector<uint8_t> data = {
        static_cast<uint8_t>(val & 0xFF),
        static_cast<uint8_t>((val >> 8) & 0xFF)
    };
    ChunkedBuf buf(data, 1);
    std::istream in(&buf);
    RawPcmDecoder dec(in, {16000, 1, 4});
    std::vector<float> out;
    std::string err;

    CHECK(dec.read_block(out, err));
    CHECK(out.size() == 1);
    CHECK_NEAR(out[0], 1234.0f / 32768.0f, 1e-6f);
    return true;
}

static bool test_stereo_frame_split_across_reads() {
    // Stereo: 1 frame = 4 bytes.  chunk=2 splits each frame across 2
    // underflow() calls (L bytes on first, R bytes on second).
    const int16_t L = 500, R = 700;
    const std::vector<uint8_t> data = {
        static_cast<uint8_t>(L & 0xFF), static_cast<uint8_t>((L >> 8) & 0xFF),
        static_cast<uint8_t>(R & 0xFF), static_cast<uint8_t>((R >> 8) & 0xFF)
    };
    ChunkedBuf buf(data, 2);
    std::istream in(&buf);
    RawPcmDecoder dec(in, {16000, 2, 4});
    std::vector<float> out;
    std::string err;

    CHECK(dec.read_block(out, err));
    CHECK(out.size() == 1);
    const float expected = (500.0f + 700.0f) / 65536.0f;
    CHECK_NEAR(out[0], expected, 1e-6f);
    return true;
}

static bool test_odd_trailing_byte_at_eof() {
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
    CHECK(r1 == true);
    CHECK(out.size() == 1);
    CHECK(err.empty());
    CHECK_NEAR(out[0], 42.0f / 32768.0f, 1e-6f);
    CHECK(dec.counters().malformed_bytes == 1);  // already counted

    // Second call: deferred error is delivered.
    const bool r2 = dec.read_block(out, err);
    CHECK(r2 == false);
    CHECK(out.empty());
    CHECK(!err.empty());
    CHECK(dec.counters().malformed_bytes == 1);  // unchanged
    return true;
}

static bool test_incomplete_stereo_frame_at_eof() {
    // 2 bytes: half a stereo frame (needs 4).  No complete frame decoded,
    // so the error is immediate.
    const std::vector<uint8_t> data = {0x01, 0x00};
    ChunkedBuf buf(data, 4096);
    std::istream in(&buf);
    RawPcmDecoder dec(in, {16000, 2, 4});
    std::vector<float> out;
    std::string err;

    const bool r = dec.read_block(out, err);
    CHECK(r == false);
    CHECK(out.empty());
    CHECK(!err.empty());
    CHECK(dec.counters().malformed_bytes == 2);
    CHECK(dec.counters().channel_frames  == 0);
    return true;
}

static bool test_invalid_sample_rate() {
    ChunkedBuf buf({}, 1);
    std::istream in(&buf);
    bool threw = false;

    // Zero
    threw = false;
    try { RawPcmDecoder dec(in, {0, 1}); }
    catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);

    // Negative
    threw = false;
    try { RawPcmDecoder dec(in, {-44100, 1}); }
    catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);

    // Unreasonably high (above kMaxSampleRate)
    threw = false;
    try { RawPcmDecoder dec(in, {RawPcmDecoder::kMaxSampleRate + 1, 1}); }
    catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);

    // Valid boundary: minimum (1 Hz)
    threw = false;
    try { RawPcmDecoder dec(in, {1, 1}); }
    catch (const std::invalid_argument&) { threw = true; }
    CHECK(!threw);

    // Valid boundary: maximum
    threw = false;
    try { RawPcmDecoder dec(in, {RawPcmDecoder::kMaxSampleRate, 1}); }
    catch (const std::invalid_argument&) { threw = true; }
    CHECK(!threw);

    return true;
}

static bool test_invalid_channel_count() {
    ChunkedBuf buf({}, 1);
    std::istream in(&buf);
    bool threw = false;

    threw = false;
    try { RawPcmDecoder dec(in, {16000, 0}); }
    catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);

    threw = false;
    try { RawPcmDecoder dec(in, {16000, 3}); }
    catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);

    threw = false;
    try { RawPcmDecoder dec(in, {16000, -1}); }
    catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);

    // Valid: 1 and 2
    threw = false;
    try { RawPcmDecoder dec(in, {16000, 1}); }
    catch (const std::invalid_argument&) { threw = true; }
    CHECK(!threw);

    threw = false;
    try { RawPcmDecoder dec(in, {16000, 2}); }
    catch (const std::invalid_argument&) { threw = true; }
    CHECK(!threw);

    return true;
}

static bool test_counter_accuracy_mono() {
    // 6 mono samples, block_frames=3 -> 2 blocks then clean EOF.
    auto data = s16le({10, 20, 30, 40, 50, 60});
    ChunkedBuf buf(data, 4096);
    std::istream in(&buf);
    RawPcmDecoder dec(in, {16000, 1, 3});

    std::vector<float> out;
    std::string err;

    CHECK(dec.read_block(out, err));
    CHECK(out.size() == 3);
    CHECK(dec.read_block(out, err));
    CHECK(out.size() == 3);
    CHECK(!dec.read_block(out, err));
    CHECK(err.empty());

    const auto& c = dec.counters();
    CHECK(c.bytes_read           == 12);
    CHECK(c.input_samples        == 6);
    CHECK(c.channel_frames       == 6);
    CHECK(c.mono_samples_emitted == 6);
    CHECK(c.malformed_bytes      == 0);
    return true;
}

static bool test_counter_accuracy_stereo() {
    // 3 stereo frames.  input_samples = 6, channel_frames = 3, emitted = 3.
    auto data = s16le({100, -100, 200, -200, 300, -300});
    ChunkedBuf buf(data, 4096);
    std::istream in(&buf);
    RawPcmDecoder dec(in, {44100, 2, 8});

    std::vector<float> all;
    std::string err;
    CHECK(decode_all(dec, all, err));
    CHECK(all.size() == 3);

    const auto& c = dec.counters();
    CHECK(c.bytes_read           == 12);   // 3 * 2ch * 2 bytes
    CHECK(c.input_samples        == 6);    // 3 * 2ch
    CHECK(c.channel_frames       == 3);
    CHECK(c.mono_samples_emitted == 3);
    CHECK(c.malformed_bytes      == 0);
    return true;
}

static bool test_internal_buffer_bounded() {
    // Decode a large stereo stream with a non-aligned chunk size to stress the
    // hold_ accumulation logic.  Verify that counters are fully consistent:
    // because hold_ is always reset to zero after each complete frame, any
    // bytes remaining at clean EOF would appear as malformed_bytes — so
    // malformed_bytes == 0 proves hold_ was correctly drained.
    //
    // hold_ is std::array<uint8_t, RawPcmDecoder::kMaxHoldBytes> — 3 bytes.
    // Its size is a compile-time constant; it does not grow with stream length.
    static_assert(RawPcmDecoder::kMaxHoldBytes == 3,
                  "bound proof relies on kMaxHoldBytes == 3");

    constexpr int N = 5000; // stereo frames
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
    CHECK(decode_all(dec, all, err));
    CHECK(err.empty());
    CHECK(all.size() == static_cast<size_t>(N));

    const auto& c = dec.counters();
    CHECK(c.bytes_read           == static_cast<uint64_t>(N * 4));
    CHECK(c.input_samples        == static_cast<uint64_t>(N * 2));
    CHECK(c.channel_frames       == static_cast<uint64_t>(N));
    CHECK(c.mono_samples_emitted == static_cast<uint64_t>(N));
    CHECK(c.malformed_bytes      == 0);  // proves hold_ was fully drained
    return true;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
    std::cout << "=== raw_pcm_stream unit tests ===\n";

    run("empty clean stream",                test_empty_clean_eof);
    run("mono min/max/zero/representative",  test_mono_min_max_zero_representative);
    run("stereo normal decoding",            test_stereo_normal_decoding);
    run("stereo downmix no overflow",        test_stereo_downmix_no_overflow);
    run("multiple output blocks",            test_multiple_output_blocks);
    run("one-byte underlying reads",         test_one_byte_underlying_reads);
    run("three-byte underlying reads",       test_three_byte_underlying_reads);
    run("five-byte underlying reads",        test_five_byte_underlying_reads);
    run("int16 split across reads",          test_int16_split_across_reads);
    run("stereo frame split across reads",   test_stereo_frame_split_across_reads);
    run("odd trailing byte at EOF",          test_odd_trailing_byte_at_eof);
    run("incomplete stereo frame at EOF",    test_incomplete_stereo_frame_at_eof);
    run("invalid sample rate",               test_invalid_sample_rate);
    run("invalid channel count",             test_invalid_channel_count);
    run("counter accuracy mono",             test_counter_accuracy_mono);
    run("counter accuracy stereo",           test_counter_accuracy_stereo);
    run("internal buffer bounded",           test_internal_buffer_bounded);

    std::cout << "\n" << g_passed << " passed, " << g_failed << " failed.\n";
    return (g_failed == 0) ? 0 : 1;
}
