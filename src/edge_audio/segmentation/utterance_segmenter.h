/// @file
/// @brief Bounded-memory endpointing of a live stream into utterances.
/// @ingroup segmentation

#pragma once

#include <cstdint>
#include <deque>
#include <vector>

#include "edge_audio/segmentation/vad.h"

namespace edge_audio {

/// @addtogroup segmentation
/// @{

/// @brief Why an utterance was closed.
enum class SegmentReason {
    end_silence,    ///< Trailing silence reached SegmenterConfig::end_silence_ms.
    max_duration,   ///< The utterance reached SegmenterConfig::max_utterance_ms.
    end_of_stream,  ///< StreamingUtteranceSegmenter::flush() closed an active utterance.
};

/// @brief A completed utterance.
///
/// Frame indices count every frame ever pushed, including pre-roll, and are
/// inclusive.
struct Utterance {
    std::vector<float> samples;  ///< Mono PCM: pre-roll + speech + retained silence.
    int sample_rate = 0;         ///< Hz.
    uint32_t id = 0;             ///< Monotonic, starting at 1.
    SegmentReason reason = SegmentReason::end_silence;  ///< Why it was closed.
    int total_frames = 0;                               ///< Frames in #samples.
    int speech_frames = 0;                              ///< Frames classified as speech.
    int64_t start_stream_frame = 0;                     ///< Stream index of the first frame.
    int64_t end_stream_frame = 0;                       ///< Stream index of the last frame.
};

/// @brief Endpointing parameters.
///
/// The constructor of StreamingUtteranceSegmenter rejects configurations where
/// `sample_rate * frame_ms` is not a multiple of 1000, any duration is shorter
/// than one frame, or the pre-roll cannot hold `speech_start_frames - 1` frames.
struct SegmenterConfig {
    int sample_rate = 16000;      ///< Input rate in Hz (no resampling is done).
    int frame_ms = 20;            ///< Decision frame length.
    int pre_roll_ms = 250;        ///< Audio kept before the detected start.
    int speech_start_frames = 2;  ///< Consecutive speech frames that open an utterance.
    int end_silence_ms = 600;     ///< Consecutive silence that closes an utterance.
    int min_speech_ms = 200;      ///< Utterances with less detected speech are discarded.
    int max_utterance_ms = 8000;  ///< Hard cap; the utterance is closed when reached.
    VadConfig vad;                ///< Frame classifier used by push().
};

/// @brief Counters exposed for telemetry and tests.
struct SegmenterTelemetry {
    uint64_t processed_frames = 0;            ///< Complete frames consumed.
    uint64_t finalized_utterances = 0;        ///< Utterances emitted.
    uint64_t discarded_short_candidates = 0;  ///< Candidates below `min_speech_ms`.
    int buffered_samples = 0;                 ///< Samples currently held.
    int max_buffered_samples_seen = 0;        ///< High-water mark of #buffered_samples.
};

/// @brief Streaming endpointer: IDLE → SPEECH → ENDING → IDLE.
///
/// Audio is pushed in arbitrary-sized blocks and classified per frame. An
/// utterance opens after `speech_start_frames` consecutive speech frames
/// (prepending up to `pre_roll_ms` of earlier audio), and closes after
/// `end_silence_ms` of silence, at `max_utterance_ms`, or on flush().
///
/// Memory is bounded: the partial-frame buffer holds less than one frame, the
/// pre-roll at most `pre_roll_ms`, and the utterance at most `max_utterance_ms`.
///
/// Not thread-safe; use from one thread.
///
/// @see docs/architecture.md, "Utterance segmentation".
class StreamingUtteranceSegmenter {
public:
    /// @throws std::invalid_argument for an inconsistent SegmenterConfig.
    explicit StreamingUtteranceSegmenter(SegmenterConfig cfg);

    /// @brief Push mono samples of any length; frames are classified with
    /// vad_frame_is_speech(). Completed utterances are queued for pop_utterances().
    void push(const float* samples, int n);

    /// @brief Push exactly one frame with an externally supplied speech decision.
    ///
    /// Bypasses the VAD so the state machine can be tested deterministically,
    /// or driven by a different frame classifier.
    /// @throws std::invalid_argument if @p n != frame_samples().
    void push_frame(const float* frame, int n, bool is_speech);

    /// @brief End of stream: emit the active utterance if it has enough speech.
    /// Idempotent. A trailing partial frame is dropped, not zero-padded.
    void flush();

    /// @brief Take all completed utterances, oldest first.
    std::vector<Utterance> pop_utterances();

    const SegmenterTelemetry& telemetry() const noexcept { return tel_; }  ///< Counters.
    int frame_samples() const noexcept { return frame_samples_; }          ///< Samples per frame.
    int pre_roll_frames() const noexcept { return pre_roll_frames_; }      ///< Pre-roll capacity.
    int max_utterance_frames() const noexcept { return max_utterance_frames_; }  ///< Utterance cap.
    const SegmenterConfig& config() const noexcept { return cfg_; }              ///< Active config.

private:
    enum class State { IDLE, SPEECH, ENDING };

    struct FrameInfo {
        std::vector<float> samples;
        bool is_speech;
    };

    void process_frame(const float* frame, bool is_speech);
    void finalize(SegmentReason reason);
    void sync_buffered_telemetry();

    int frame_samples_;
    int pre_roll_frames_;
    int end_silence_frames_;
    int min_speech_frames_;
    int max_utterance_frames_;

    SegmenterConfig cfg_;
    State state_ = State::IDLE;
    int consecutive_speech_ = 0;
    int consecutive_silence_ = 0;
    int speech_frames_in_utt_ = 0;
    int64_t stream_frame_idx_ = 0;
    int64_t utt_start_frame_ = 0;
    uint32_t next_id_ = 1;
    bool flushed_ = false;

    std::deque<FrameInfo> pre_roll_;  // IDLE only; <= pre_roll_frames_
    std::vector<float> utterance_;    // SPEECH/ENDING; <= max_utterance_frames_
    std::vector<float> partial_;      // < frame_samples_
    std::vector<Utterance> ready_;
    SegmenterTelemetry tel_;
};

/// @}

}  // namespace edge_audio
