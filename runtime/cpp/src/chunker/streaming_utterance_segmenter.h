#pragma once

#include <cstdint>
#include <deque>
#include <stdexcept>
#include <string>
#include <vector>

#include "chunker/vad.h"

namespace pipeline {

// -----------------------------------------------------------------------
// Reason an utterance was closed.
// -----------------------------------------------------------------------
enum class SegmentReason {
    end_silence,    // consecutive silence frames reached end_silence_ms
    max_duration,   // appended frame count reached max_utterance_ms
    end_of_stream,  // flush() called while minimum speech was met
};

// -----------------------------------------------------------------------
// A completed utterance emitted by StreamingUtteranceSegmenter.
//
// start_stream_frame / end_stream_frame are zero-based indices of the
// first and last 20 ms frame in the stream (inclusive, counting from the
// very first frame ever fed, including pre-roll frames).
// -----------------------------------------------------------------------
struct Utterance {
    std::vector<float> samples;           // mono float32 PCM at sample_rate
    int          sample_rate       = 0;
    uint32_t     id                = 0;   // monotonically increasing, starts at 1
    SegmentReason reason           = SegmentReason::end_silence;
    int          total_frames      = 0;   // pre-roll + speech + retained silence
    int          speech_frames     = 0;   // frames classified as speech
    int64_t      start_stream_frame = 0;  // inclusive
    int64_t      end_stream_frame   = 0;  // inclusive
};

// -----------------------------------------------------------------------
// Configuration for StreamingUtteranceSegmenter.
//
// Constraints enforced in the constructor (throw std::invalid_argument):
//   - sample_rate > 0
//   - frame_ms > 0
//   - sample_rate * frame_ms divisible by 1000
//   - speech_start_frames >= 1
//   - end_silence_ms > 0
//   - max_utterance_ms > 0
//   - end_silence_ms / frame_ms >= 1
//   - max_utterance_ms / frame_ms >= 1
//   - pre_roll_ms / frame_ms >= speech_start_frames - 1
// -----------------------------------------------------------------------
struct SegmenterConfig {
    int sample_rate         = 16000;  // input Hz; metadata only, no resampling
    int frame_ms            = 20;     // frame length in ms
    int pre_roll_ms         = 250;    // frames prepended before speech start
    int speech_start_frames = 2;      // consecutive speech frames to start utterance
    int end_silence_ms      = 600;    // consecutive silence that closes utterance
    int min_speech_ms       = 200;    // minimum detected speech to emit (else discard)
    int max_utterance_ms    = 8000;   // hard cap; finalize immediately when reached

    // Shared VAD frame classifier parameters (used by production push() path).
    VadConfig vad;
};

// -----------------------------------------------------------------------
// Telemetry counters.
// -----------------------------------------------------------------------
struct SegmenterTelemetry {
    uint64_t processed_frames           = 0;  // complete frames consumed
    uint64_t finalized_utterances       = 0;  // utterances emitted
    uint64_t discarded_short_candidates = 0;  // utterances dropped (< min_speech_ms)
    int      buffered_samples           = 0;  // current samples held in all buffers
    int      max_buffered_samples_seen  = 0;  // peak buffered_samples ever observed
};

// -----------------------------------------------------------------------
// StreamingUtteranceSegmenter
//
// State machine: IDLE -> SPEECH -> ENDING -> IDLE
//
// Buffer bounds:
//   partial_buf  : < frame_samples samples
//   pre_roll_buf : <= (pre_roll_ms / frame_ms) complete frames
//   utterance_buf: <= (max_utterance_ms / frame_ms) complete frames
//
// Thread safety: none; single-threaded use only.
// -----------------------------------------------------------------------
class StreamingUtteranceSegmenter {
public:
    explicit StreamingUtteranceSegmenter(SegmenterConfig cfg);

    // Production path: push arbitrary-length mono float32 samples.
    // Accumulates partial frames internally; calls vad_frame_is_speech per
    // complete frame.  Completed utterances are queued for pop_utterances().
    void push(const float* samples, int n);

    // Deterministic test path: inject one complete frame with an explicit
    // speech/non-speech decision.  n must equal frame_samples().
    // Bypasses vad_frame_is_speech; used for state-machine unit tests.
    void push_frame(const float* frame, int n, bool is_speech);

    // Signal end of stream.  Idempotent; safe to call multiple times.
    // Does not zero-pad an incomplete partial frame.
    // Emits the active utterance only if minimum detected speech was reached.
    void flush();

    // Consume all completed utterances.  Clears the internal ready queue.
    std::vector<Utterance> pop_utterances();

    const SegmenterTelemetry& telemetry() const noexcept { return tel_; }

    int frame_samples() const noexcept { return frame_samples_; }
    int pre_roll_frames() const noexcept { return pre_roll_frames_; }
    int max_utterance_frames() const noexcept { return max_utterance_frames_; }
    const SegmenterConfig& config() const noexcept { return cfg_; }

private:
    enum class State { IDLE, SPEECH, ENDING };

    // Derived frame counts (computed once in constructor).
    int frame_samples_;
    int pre_roll_frames_;
    int end_silence_frames_;
    int min_speech_frames_;
    int max_utterance_frames_;

    SegmenterConfig cfg_;
    State    state_               = State::IDLE;
    int      consecutive_speech_  = 0;  // IDLE: speech frames since last reset
    int      consecutive_silence_ = 0;  // ENDING: silence frames since speech ended
    int      speech_frames_in_utt_ = 0;
    int64_t  stream_frame_idx_    = 0;  // total frames processed so far
    int64_t  utt_start_frame_     = 0;  // stream index of first utterance frame
    uint32_t next_id_             = 1;
    bool     flushed_             = false;

    // Each entry holds one complete frame and its VAD classification.
    struct FrameInfo {
        std::vector<float> samples;
        bool               is_speech;
    };

    std::deque<FrameInfo>  pre_roll_;   // IDLE only; bounded by pre_roll_frames_
    std::vector<float>     utterance_;  // SPEECH/ENDING; bounded by max_utterance_frames_
    std::vector<float>     partial_;    // accumulates sub-frame samples; < frame_samples_

    std::vector<Utterance> ready_;      // completed utterances awaiting pop

    SegmenterTelemetry tel_;

    // Process one complete frame (both paths converge here).
    void process_frame(const float* frame, bool is_speech);

    // Close the active utterance.  Emits or discards based on min_speech_ms.
    // Resets to IDLE.
    void finalize(SegmentReason reason);

    // Recompute tel_.buffered_samples and update max_buffered_samples_seen.
    void sync_buffered_telemetry();
};

}  // namespace pipeline
