// StreamingUtteranceSegmenter state machine, boundaries and memory bounds.

#include <cmath>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "edge_audio/segmentation/utterance_segmenter.h"

using namespace edge_audio;

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

// Default config: 16 kHz, 20 ms frames, default thresholds.
static SegmenterConfig default_cfg() {
    SegmenterConfig c;
    // Defaults from header: 16000 Hz, 20ms, 250ms preroll, 2 start frames,
    // 600ms end silence, 200ms min speech, 8000ms max utterance.
    return c;
}

// frame_samples for default config
static constexpr int kRate = 16000;
static constexpr int kFrameMs = 20;
static constexpr int kFS = kRate * kFrameMs / 1000;  // 320

// Push N complete frames via push_frame.  All frames use the given is_speech.
static void push_n(StreamingUtteranceSegmenter& seg, int n, bool is_speech) {
    static const std::vector<float> kZero(kFS, 0.0f);
    for (int i = 0; i < n; ++i) {
        seg.push_frame(kZero.data(), kFS, is_speech);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// 1. Silence produces nothing.
TEST(UtteranceSegmenter, SilenceProducesNothing) {
    StreamingUtteranceSegmenter seg(default_cfg());
    push_n(seg, 100, false);
    seg.flush();
    auto utts = seg.pop_utterances();
    ASSERT_TRUE(utts.empty());
    ASSERT_EQ(seg.telemetry().processed_frames, 100);
    ASSERT_EQ(seg.telemetry().finalized_utterances, 0);
    ASSERT_EQ(seg.telemetry().discarded_short_candidates, 0);
}

// 2. Start trigger fires after speech_start_frames consecutive speech frames;
//    pre-roll frames are prepended.
TEST(UtteranceSegmenter, StartTriggerAndPreroll) {
    SegmenterConfig cfg = default_cfg();
    // Disable min_speech filter so this test focuses only on pre-roll semantics.
    cfg.min_speech_ms = 0;
    // pre_roll_frames = 250/20 = 12, speech_start_frames = 2
    StreamingUtteranceSegmenter seg(cfg);

    const int pre_roll_frames = cfg.pre_roll_ms / cfg.frame_ms;  // 12
    const int start_frames = cfg.speech_start_frames;            // 2

    // Push enough silence to saturate the pre-roll ring.
    push_n(seg, pre_roll_frames + 4, false);  // 16 silence frames

    // Push exactly speech_start_frames speech frames to trigger.
    push_n(seg, start_frames, true);

    // Use flush to close.
    seg.flush();

    auto utts = seg.pop_utterances();
    ASSERT_EQ(utts.size(), 1u);

    const Utterance& u = utts[0];
    // At trigger time the pre-roll ring holds pre_roll_frames frames
    // (including the first speech frame that did not yet fire the trigger).
    // The trigger frame itself is appended exactly once.
    // Total = pre_roll_frames (flushed ring) + 1 (trigger frame).
    const int expected_total = pre_roll_frames + 1;
    ASSERT_EQ(u.total_frames, expected_total);
    ASSERT_EQ(u.sample_rate, kRate);
    ASSERT_EQ(u.reason, SegmentReason::end_of_stream);
    // speech_frames: one speech frame inside the ring (the first speech frame)
    //                plus the trigger frame = start_frames total.
    ASSERT_EQ(u.speech_frames, start_frames);
    // Samples length
    ASSERT_EQ(static_cast<int>(u.samples.size()), expected_total * kFS);
}

// 3. Split input frame: pushing half a frame then the other half is the same
//    as one full push.
TEST(UtteranceSegmenter, SplitInputFrame) {
    // Feed only silence so we can verify frame counting without VAD.
    // Two configs produce the same processed_frames count.

    SegmenterConfig cfg = default_cfg();
    const int half = kFS / 2;
    const std::vector<float> silence(kFS, 0.0f);

    // Segmenter A: push 10 full silence frames
    {
        StreamingUtteranceSegmenter a(cfg);
        for (int i = 0; i < 10; ++i) a.push(silence.data(), kFS);
        a.flush();
        ASSERT_EQ(a.telemetry().processed_frames, 10u);
        ASSERT_TRUE(a.pop_utterances().empty());
    }

    // Segmenter B: push the same data split across two calls per frame
    {
        StreamingUtteranceSegmenter b(cfg);
        for (int i = 0; i < 10; ++i) {
            b.push(silence.data(), half);
            b.push(silence.data() + half, kFS - half);
        }
        b.flush();
        ASSERT_EQ(b.telemetry().processed_frames, 10u);
        ASSERT_TRUE(b.pop_utterances().empty());
    }

    // Segmenter C: one big push with partial trailing bytes
    {
        StreamingUtteranceSegmenter c(cfg);
        const std::vector<float> buf(10 * kFS + half, 0.0f);
        c.push(buf.data(), static_cast<int>(buf.size()));
        // 10 complete frames; half frame is in partial buffer
        ASSERT_EQ(c.telemetry().processed_frames, 10u);
        c.flush();  // partial discarded
        ASSERT_TRUE(c.pop_utterances().empty());
    }
}

// 4. Exact end-silence boundary: exactly end_silence_frames non-speech frames
//    after speech closes the utterance.
TEST(UtteranceSegmenter, ExactEndSilenceBoundary) {
    SegmenterConfig cfg = default_cfg();
    // end_silence_frames = 600/20 = 30, min_speech_frames = 200/20 = 10
    StreamingUtteranceSegmenter seg(cfg);

    const int end_silence_frames = cfg.end_silence_ms / cfg.frame_ms;  // 30
    const int min_speech_frames = cfg.min_speech_ms / cfg.frame_ms;    // 10
    const int start_frames = cfg.speech_start_frames;                  // 2
    const int pre_roll_frames = cfg.pre_roll_ms / cfg.frame_ms;        // 12

    // Trigger the utterance.
    push_n(seg, pre_roll_frames, false);
    push_n(seg, start_frames, true);

    // Continue speech until min is satisfied.
    push_n(seg, min_speech_frames - start_frames + 1, true);

    // Push exactly end_silence_frames non-speech frames.
    push_n(seg, end_silence_frames, false);

    // Utterance should be emitted with reason end_silence.
    auto utts = seg.pop_utterances();
    ASSERT_EQ(utts.size(), 1u);
    ASSERT_EQ(utts[0].reason, SegmentReason::end_silence);

    // State is back to IDLE; no active utterance.
    seg.flush();
    ASSERT_TRUE(seg.pop_utterances().empty());
}

// 5. Resumed speech in ENDING does not split the utterance.
TEST(UtteranceSegmenter, ResumedSpeechDoesNotSplit) {
    SegmenterConfig cfg = default_cfg();
    StreamingUtteranceSegmenter seg(cfg);

    const int end_silence_frames = cfg.end_silence_ms / cfg.frame_ms;  // 30
    const int min_speech_frames = cfg.min_speech_ms / cfg.frame_ms;    // 10
    const int start_frames = cfg.speech_start_frames;
    const int pre_roll_frames = cfg.pre_roll_ms / cfg.frame_ms;

    // Start utterance.
    push_n(seg, pre_roll_frames, false);
    push_n(seg, start_frames + min_speech_frames, true);

    // Push end_silence_frames - 1 silence (below threshold).
    push_n(seg, end_silence_frames - 1, false);

    // Resume speech before silence threshold -> no split.
    push_n(seg, min_speech_frames, true);

    // No utterance should have been emitted yet.
    ASSERT_TRUE(seg.pop_utterances().empty());

    // Close cleanly.
    seg.flush();
    auto utts = seg.pop_utterances();
    ASSERT_EQ(utts.size(), 1u);
    ASSERT_EQ(utts[0].reason, SegmentReason::end_of_stream);
}

// 6. Short candidate (< min_speech_ms) is discarded and counted.
TEST(UtteranceSegmenter, ShortCandidateDiscarded) {
    SegmenterConfig cfg = default_cfg();
    // min_speech_frames = 10; we'll provide only 2 speech frames.
    StreamingUtteranceSegmenter seg(cfg);

    const int end_silence_frames = cfg.end_silence_ms / cfg.frame_ms;
    const int pre_roll_frames = cfg.pre_roll_ms / cfg.frame_ms;
    const int start_frames = cfg.speech_start_frames;  // 2

    // Trigger with minimal speech (exactly start_frames = 2, below min_speech_frames=10).
    push_n(seg, pre_roll_frames, false);
    push_n(seg, start_frames, true);  // 2 speech frames total

    // End immediately with silence.
    push_n(seg, end_silence_frames, false);

    // Candidate should be discarded.
    auto utts = seg.pop_utterances();
    ASSERT_TRUE(utts.empty());
    ASSERT_EQ(seg.telemetry().discarded_short_candidates, 1u);
    ASSERT_EQ(seg.telemetry().finalized_utterances, 0u);
}

// 7. Max-duration finalization: utterance closed with reason max_duration.
TEST(UtteranceSegmenter, MaxDurationFinalization) {
    SegmenterConfig cfg = default_cfg();
    // max_utterance_frames = 8000/20 = 400
    StreamingUtteranceSegmenter seg(cfg);

    const int max_frames = cfg.max_utterance_ms / cfg.frame_ms;  // 400
    const int pre_roll_frames = cfg.pre_roll_ms / cfg.frame_ms;  // 12
    const int start_frames = cfg.speech_start_frames;            // 2

    // Trigger.
    push_n(seg, pre_roll_frames, false);
    push_n(seg, start_frames, true);

    // At trigger time the utterance holds pre_roll_frames (flushed ring)
    // + 1 (trigger frame only; the first speech frame is inside the ring).
    const int already = pre_roll_frames + 1;
    push_n(seg, max_frames - already, true);

    auto utts = seg.pop_utterances();
    ASSERT_EQ(utts.size(), 1u);
    ASSERT_EQ(utts[0].reason, SegmentReason::max_duration);
    ASSERT_EQ(utts[0].total_frames, max_frames);
}

// 8. flush() emits an active utterance; repeated flush() is idempotent.
TEST(UtteranceSegmenter, EofAndRepeatedEof) {
    SegmenterConfig cfg = default_cfg();
    StreamingUtteranceSegmenter seg(cfg);

    const int pre_roll_frames = cfg.pre_roll_ms / cfg.frame_ms;
    const int start_frames = cfg.speech_start_frames;
    const int min_speech_frames = cfg.min_speech_ms / cfg.frame_ms;

    // Start a valid utterance.
    push_n(seg, pre_roll_frames, false);
    push_n(seg, start_frames + min_speech_frames, true);

    // First flush: should emit.
    seg.flush();
    auto utts = seg.pop_utterances();
    ASSERT_EQ(utts.size(), 1u);
    ASSERT_EQ(utts[0].reason, SegmentReason::end_of_stream);

    // Second flush: idempotent, nothing new.
    seg.flush();
    ASSERT_TRUE(seg.pop_utterances().empty());

    // Third flush: still idempotent.
    seg.flush();
    ASSERT_TRUE(seg.pop_utterances().empty());

    // Counters unchanged after extra flushes.
    ASSERT_EQ(seg.telemetry().finalized_utterances, 1u);
}

// 9. Two separated utterances produce two outputs with distinct IDs.
TEST(UtteranceSegmenter, TwoSeparatedUtterances) {
    SegmenterConfig cfg = default_cfg();
    StreamingUtteranceSegmenter seg(cfg);

    const int end_silence_frames = cfg.end_silence_ms / cfg.frame_ms;  // 30
    const int min_speech_frames = cfg.min_speech_ms / cfg.frame_ms;    // 10
    const int start_frames = cfg.speech_start_frames;
    const int pre_roll_frames = cfg.pre_roll_ms / cfg.frame_ms;

    // First utterance.
    push_n(seg, pre_roll_frames, false);
    push_n(seg, start_frames + min_speech_frames, true);
    push_n(seg, end_silence_frames, false);

    {
        auto utts = seg.pop_utterances();
        ASSERT_EQ(utts.size(), 1u);
        ASSERT_EQ(utts[0].id, 1u);
        ASSERT_EQ(utts[0].reason, SegmentReason::end_silence);
    }

    // Second utterance (machine is back in IDLE; silence resets the ring).
    push_n(seg, pre_roll_frames, false);
    push_n(seg, start_frames + min_speech_frames, true);
    seg.flush();

    {
        auto utts = seg.pop_utterances();
        ASSERT_EQ(utts.size(), 1u);
        ASSERT_EQ(utts[0].id, 2u);
        ASSERT_EQ(utts[0].reason, SegmentReason::end_of_stream);
    }
}

// 10. IDs increase monotonically; counters accumulate.
TEST(UtteranceSegmenter, IncreasingIdsAndCounters) {
    SegmenterConfig cfg = default_cfg();
    StreamingUtteranceSegmenter seg(cfg);

    const int end_silence_frames = cfg.end_silence_ms / cfg.frame_ms;
    const int min_speech_frames = cfg.min_speech_ms / cfg.frame_ms;
    const int start_frames = cfg.speech_start_frames;
    const int pre_roll_frames = cfg.pre_roll_ms / cfg.frame_ms;

    const int N = 3;
    for (int k = 0; k < N; ++k) {
        push_n(seg, pre_roll_frames, false);
        push_n(seg, start_frames + min_speech_frames, true);
        push_n(seg, end_silence_frames, false);
    }

    auto utts = seg.pop_utterances();
    ASSERT_EQ(static_cast<int>(utts.size()), N);
    for (int k = 0; k < N; ++k) {
        ASSERT_EQ(utts[k].id, static_cast<uint32_t>(k + 1));
    }
    ASSERT_EQ(seg.telemetry().finalized_utterances, static_cast<uint64_t>(N));
}

// 11. Incomplete final frame is silently discarded at EOF.
TEST(UtteranceSegmenter, IncompleteFinalFrame) {
    SegmenterConfig cfg = default_cfg();
    StreamingUtteranceSegmenter seg(cfg);

    // Push 5 full silence frames, then kFS/2 extra samples (incomplete frame).
    const std::vector<float> silence(kFS, 0.0f);
    for (int i = 0; i < 5; ++i) seg.push(silence.data(), kFS);
    seg.push(silence.data(), kFS / 2);

    ASSERT_EQ(seg.telemetry().processed_frames, 5u);

    seg.flush();  // partial discarded, no utterance
    ASSERT_TRUE(seg.pop_utterances().empty());
    ASSERT_EQ(seg.telemetry().processed_frames, 5u);  // unchanged after flush
}

// 12. Buffered samples stay within declared bounds.
TEST(UtteranceSegmenter, BoundedPrerollAndUtteranceStorage) {
    SegmenterConfig cfg = default_cfg();
    StreamingUtteranceSegmenter seg(cfg);

    const int pre_roll_frames = seg.pre_roll_frames();            // 12
    const int max_utterance_frames = seg.max_utterance_frames();  // 400
    const int frame_samples = seg.frame_samples();                // 320

    // Bound: partial (<frame_samples) + pre_roll OR utterance (never both).
    const int max_allowed =
        (pre_roll_frames + max_utterance_frames) * frame_samples + (frame_samples - 1);

    // Phase 1: fill pre-roll (IDLE).
    push_n(seg, pre_roll_frames + 10, false);
    ASSERT_LE(seg.telemetry().buffered_samples, max_allowed);

    // Phase 2: trigger and saturate utterance (SPEECH).
    const int start_frames = cfg.speech_start_frames;
    push_n(seg, start_frames, true);
    push_n(seg, max_utterance_frames, true);  // will hit max_duration and reset

    ASSERT_LE(seg.telemetry().buffered_samples, max_allowed);
    ASSERT_LE(seg.telemetry().max_buffered_samples_seen, max_allowed);

    seg.flush();
}

// 13. Invalid configurations throw std::invalid_argument.
TEST(UtteranceSegmenter, InvalidConfigurations) {
    auto make = [](SegmenterConfig c) -> bool {
        try {
            StreamingUtteranceSegmenter seg(c);
        } catch (const std::invalid_argument&) {
            return true;
        }
        return false;  // should have thrown
    };

    SegmenterConfig bad;

    // sample_rate <= 0
    bad = default_cfg();
    bad.sample_rate = 0;
    ASSERT_TRUE(make(bad));

    bad = default_cfg();
    bad.sample_rate = -1;
    ASSERT_TRUE(make(bad));

    // frame_ms <= 0
    bad = default_cfg();
    bad.frame_ms = 0;
    ASSERT_TRUE(make(bad));

    // sample_rate * frame_ms not divisible by 1000
    bad = default_cfg();
    bad.sample_rate = 16001;
    bad.frame_ms = 20;
    ASSERT_TRUE(make(bad));

    // speech_start_frames < 1
    bad = default_cfg();
    bad.speech_start_frames = 0;
    ASSERT_TRUE(make(bad));

    // end_silence_ms <= 0
    bad = default_cfg();
    bad.end_silence_ms = 0;
    ASSERT_TRUE(make(bad));

    // max_utterance_ms <= 0
    bad = default_cfg();
    bad.max_utterance_ms = 0;
    ASSERT_TRUE(make(bad));

    // pre_roll too small: pre_roll_ms/frame_ms < speech_start_frames - 1
    // With speech_start_frames=3, we need pre_roll_frames>=2 (40ms).
    // Setting pre_roll_ms=20ms -> 1 frame < 2: invalid.
    bad = default_cfg();
    bad.speech_start_frames = 3;
    bad.pre_roll_ms = 20;
    ASSERT_TRUE(make(bad));

    // end_silence_ms too small to yield >= 1 frame
    // frame_ms=20, end_silence_ms=19 -> 0 frames
    bad = default_cfg();
    bad.end_silence_ms = 19;
    ASSERT_TRUE(make(bad));

    // min_speech_ms < 0
    bad = default_cfg();
    bad.min_speech_ms = -1;
    ASSERT_TRUE(make(bad));
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
