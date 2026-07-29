// streaming_utterance_segmenter.cpp
// Bounded streaming utterance segmenter.
//
// State machine: IDLE -> SPEECH -> ENDING -> IDLE
//
// In IDLE the pre-roll ring holds frames preceding the current frame.
// When speech_start_frames consecutive speech frames are observed the
// accumulated pre-roll is flushed then the trigger frame is appended exactly
// once and the machine enters SPEECH.
//
// In SPEECH each frame is appended to the utterance buffer.  The first
// non-speech frame transitions to ENDING (silence count = 1).
//
// In ENDING each non-speech frame increments the silence counter; reaching
// end_silence_frames_ triggers finalization.  A speech frame resets the
// counter and returns to SPEECH.
//
// Finalization discards the candidate when speech_frames < min_speech_frames_
// and emits a completed Utterance otherwise.  Both paths reset to IDLE.
//
// Maximum duration is checked after every append in SPEECH and ENDING; the
// utterance is finalized immediately when total appended frames reaches
// max_utterance_frames_.

#include "chunker/streaming_utterance_segmenter.h"

#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <string>

namespace pipeline {

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

StreamingUtteranceSegmenter::StreamingUtteranceSegmenter(SegmenterConfig cfg)
    : cfg_(cfg)
{
    if (cfg_.sample_rate <= 0)
        throw std::invalid_argument("sample_rate must be > 0");
    if (cfg_.frame_ms <= 0)
        throw std::invalid_argument("frame_ms must be > 0");
    if ((cfg_.sample_rate * cfg_.frame_ms) % 1000 != 0)
        throw std::invalid_argument(
            "sample_rate * frame_ms must be divisible by 1000 "
            "(frame_samples must be a whole number)");
    if (cfg_.speech_start_frames < 1)
        throw std::invalid_argument("speech_start_frames must be >= 1");
    if (cfg_.end_silence_ms <= 0)
        throw std::invalid_argument("end_silence_ms must be > 0");
    if (cfg_.max_utterance_ms <= 0)
        throw std::invalid_argument("max_utterance_ms must be > 0");
    if (cfg_.min_speech_ms < 0)
        throw std::invalid_argument("min_speech_ms must be >= 0");

    frame_samples_       = cfg_.sample_rate * cfg_.frame_ms / 1000;
    pre_roll_frames_     = cfg_.pre_roll_ms / cfg_.frame_ms;
    end_silence_frames_  = cfg_.end_silence_ms / cfg_.frame_ms;
    min_speech_frames_   = cfg_.min_speech_ms / cfg_.frame_ms;
    max_utterance_frames_ = cfg_.max_utterance_ms / cfg_.frame_ms;

    if (pre_roll_frames_ < cfg_.speech_start_frames - 1)
        throw std::invalid_argument(
            "pre_roll capacity (pre_roll_ms / frame_ms) must be >= speech_start_frames - 1");
    if (end_silence_frames_ < 1)
        throw std::invalid_argument(
            "end_silence_ms / frame_ms must be >= 1 (increase end_silence_ms or decrease frame_ms)");
    if (max_utterance_frames_ < 1)
        throw std::invalid_argument(
            "max_utterance_ms / frame_ms must be >= 1");

    partial_.reserve(static_cast<size_t>(frame_samples_));
    utterance_.reserve(static_cast<size_t>(max_utterance_frames_) *
                       static_cast<size_t>(frame_samples_));
}

// ---------------------------------------------------------------------------
// Public interface
// ---------------------------------------------------------------------------

void StreamingUtteranceSegmenter::push(const float* samples, int n) {
    if (flushed_ || samples == nullptr || n <= 0) return;

    int i = 0;
    while (i < n) {
        const int need  = frame_samples_ - static_cast<int>(partial_.size());
        const int avail = n - i;
        const int take  = std::min(need, avail);

        partial_.insert(partial_.end(), samples + i, samples + i + take);
        i += take;

        if (static_cast<int>(partial_.size()) == frame_samples_) {
            const bool is_speech = vad_frame_is_speech(
                partial_.data(), frame_samples_, cfg_.sample_rate, cfg_.vad);
            process_frame(partial_.data(), is_speech);
            partial_.clear();
        }
    }
    sync_buffered_telemetry();
}

void StreamingUtteranceSegmenter::push_frame(const float* frame, int n, bool is_speech) {
    if (flushed_ || frame == nullptr) return;
    assert(n == frame_samples_ && "push_frame: n must equal frame_samples()");
    (void)n;  // suppress unused-parameter warning in release builds
    process_frame(frame, is_speech);
    sync_buffered_telemetry();
}

void StreamingUtteranceSegmenter::flush() {
    if (flushed_) return;
    flushed_ = true;
    // Partial frame: EOF never zero-pads an incomplete frame; discard it.
    partial_.clear();

    if (state_ != State::IDLE) {
        if (speech_frames_in_utt_ >= min_speech_frames_) {
            // Emit the active utterance.
            const int total_frames =
                static_cast<int>(utterance_.size()) / frame_samples_;
            Utterance u;
            u.samples           = std::move(utterance_);
            u.sample_rate       = cfg_.sample_rate;
            u.id                = next_id_++;
            u.reason            = SegmentReason::end_of_stream;
            u.total_frames      = total_frames;
            u.speech_frames     = speech_frames_in_utt_;
            u.start_stream_frame = utt_start_frame_;
            u.end_stream_frame  = stream_frame_idx_ - 1;
            ready_.push_back(std::move(u));
            ++tel_.finalized_utterances;
        } else {
            // Short candidate: discard.
            ++tel_.discarded_short_candidates;
        }
        utterance_.clear();
        speech_frames_in_utt_ = 0;
        state_                = State::IDLE;
    }
    sync_buffered_telemetry();
}

std::vector<Utterance> StreamingUtteranceSegmenter::pop_utterances() {
    std::vector<Utterance> out;
    out.swap(ready_);
    return out;
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

void StreamingUtteranceSegmenter::process_frame(const float* frame, bool is_speech) {
    ++tel_.processed_frames;
    const int64_t current_idx = stream_frame_idx_++;

    switch (state_) {

    // ------------------------------------------------------------------
    case State::IDLE: {
        if (is_speech) {
            ++consecutive_speech_;
        } else {
            consecutive_speech_ = 0;
        }

        if (consecutive_speech_ >= cfg_.speech_start_frames) {
            // Trigger: flush pre-roll then append this (trigger) frame.
            utt_start_frame_      = current_idx - static_cast<int64_t>(pre_roll_.size());
            speech_frames_in_utt_ = 0;
            utterance_.clear();

            for (auto& fi : pre_roll_) {
                utterance_.insert(utterance_.end(),
                                  fi.samples.begin(), fi.samples.end());
                if (fi.is_speech) ++speech_frames_in_utt_;
            }
            pre_roll_.clear();

            utterance_.insert(utterance_.end(), frame, frame + frame_samples_);
            ++speech_frames_in_utt_;  // trigger frame is always speech

            consecutive_speech_  = 0;
            consecutive_silence_ = 0;
            state_               = State::SPEECH;

            // Check max duration (pre-roll + trigger frame may already hit cap).
            const int n_frames =
                static_cast<int>(utterance_.size()) / frame_samples_;
            if (n_frames >= max_utterance_frames_) {
                finalize(SegmentReason::max_duration);
            }
        } else {
            // Not triggered yet: store frame in pre-roll ring.
            FrameInfo fi;
            fi.samples.assign(frame, frame + frame_samples_);
            fi.is_speech = is_speech;
            pre_roll_.push_back(std::move(fi));
            if (static_cast<int>(pre_roll_.size()) > pre_roll_frames_) {
                pre_roll_.pop_front();
            }
        }
        break;
    }

    // ------------------------------------------------------------------
    case State::SPEECH: {
        utterance_.insert(utterance_.end(), frame, frame + frame_samples_);

        if (is_speech) {
            ++speech_frames_in_utt_;
            // consecutive_silence_ already 0 here
        } else {
            // First non-speech frame: silence count = 1, enter ENDING.
            consecutive_silence_ = 1;
            state_               = State::ENDING;
            if (consecutive_silence_ >= end_silence_frames_) {
                finalize(SegmentReason::end_silence);
                return;
            }
        }

        // Max duration check (after append).
        {
            const int n_frames =
                static_cast<int>(utterance_.size()) / frame_samples_;
            if (n_frames >= max_utterance_frames_) {
                finalize(SegmentReason::max_duration);
            }
        }
        break;
    }

    // ------------------------------------------------------------------
    case State::ENDING: {
        utterance_.insert(utterance_.end(), frame, frame + frame_samples_);

        if (is_speech) {
            ++speech_frames_in_utt_;
            consecutive_silence_ = 0;
            state_               = State::SPEECH;
        } else {
            ++consecutive_silence_;
            if (consecutive_silence_ >= end_silence_frames_) {
                finalize(SegmentReason::end_silence);
                return;
            }
        }

        // Max duration check (after append).
        {
            const int n_frames =
                static_cast<int>(utterance_.size()) / frame_samples_;
            if (n_frames >= max_utterance_frames_) {
                finalize(SegmentReason::max_duration);
            }
        }
        break;
    }

    }  // switch
}

void StreamingUtteranceSegmenter::finalize(SegmentReason reason) {
    const int total_frames =
        static_cast<int>(utterance_.size()) / frame_samples_;

    if (speech_frames_in_utt_ >= min_speech_frames_) {
        Utterance u;
        u.samples            = std::move(utterance_);
        u.sample_rate        = cfg_.sample_rate;
        u.id                 = next_id_++;
        u.reason             = reason;
        u.total_frames       = total_frames;
        u.speech_frames      = speech_frames_in_utt_;
        u.start_stream_frame = utt_start_frame_;
        u.end_stream_frame   = stream_frame_idx_ - 1;
        ready_.push_back(std::move(u));
        ++tel_.finalized_utterances;
    } else {
        ++tel_.discarded_short_candidates;
    }

    // Reset utterance state; return to IDLE.
    utterance_.clear();
    utterance_.reserve(static_cast<size_t>(max_utterance_frames_) *
                       static_cast<size_t>(frame_samples_));
    speech_frames_in_utt_ = 0;
    consecutive_silence_  = 0;
    consecutive_speech_   = 0;
    pre_roll_.clear();
    state_ = State::IDLE;
}

void StreamingUtteranceSegmenter::sync_buffered_telemetry() {
    int total = static_cast<int>(partial_.size());
    for (const auto& fi : pre_roll_) {
        total += static_cast<int>(fi.samples.size());
    }
    total += static_cast<int>(utterance_.size());
    tel_.buffered_samples = total;
    if (total > tel_.max_buffered_samples_seen) {
        tel_.max_buffered_samples_seen = total;
    }
}

}  // namespace pipeline
