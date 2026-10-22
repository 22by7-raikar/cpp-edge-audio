/// @file
/// @brief Streaming voice-command pipeline: PCM in, one JSON event per utterance out.
/// @ingroup command

#pragma once

#include <string>
#include <vector>

#include "edge_audio/asr/transcription.h"
#include "edge_audio/quality/admission_policy.h"
#include "edge_audio/quality/utterance_analyzer.h"
#include "edge_audio/segmentation/utterance_segmenter.h"

namespace edge_audio {

/// @addtogroup command
/// @{

/// @brief VoiceCommandLoop settings.
struct VoiceCommandConfig {
    SegmenterConfig segmenter;                            ///< Endpointing parameters.
    QualityPolicy quality_policy = QualityPolicy::RULE;   ///< Admission policy.
    double quality_threshold = kQualityDefaultThreshold;  ///< Learned threshold.
    std::string capture_device = "unknown";               ///< Copied into every event.
    std::string transport = "stdin_pcm";                  ///< Copied into every event.
};

/// @brief Segments a PCM stream and turns each utterance into a command event.
///
/// Per utterance: UtteranceAnalyzer decides admission; if admitted the injected
/// Transcriber runs once on the whole utterance; the transcript is routed with
/// route_command(); one JSON line is queued. All work happens synchronously on
/// the caller's thread inside push()/flush(), so ASR latency delays the next
/// push — see docs/architecture.md for the threading discussion.
///
/// The Transcriber seam lets tests run the full loop without a model.
class VoiceCommandLoop {
public:
    /// @throws std::invalid_argument if @p transcribe is empty or the config is invalid.
    VoiceCommandLoop(VoiceCommandConfig config, Transcriber transcribe);

    /// @brief Feed mono samples at `config.segmenter.sample_rate`.
    void push(const float* samples, int sample_count);

    /// @brief End of stream: close and process any active utterance.
    void flush();

    /// @brief Take queued events (one JSON object per string, no newline).
    std::vector<std::string> pop_json_events();

    /// @brief Process one already-segmented utterance and return its event JSON.
    /// push() reaches the same code path; exposed for deterministic tests.
    std::string process_completed_utterance(Utterance utterance);

    /// @brief Segmenter counters.
    const SegmenterTelemetry& telemetry() const noexcept { return segmenter_.telemetry(); }

    /// @brief True if any admitted utterance failed ASR.
    bool had_asr_error() const noexcept { return had_asr_error_; }

private:
    void drain_completed();

    VoiceCommandConfig config_;
    StreamingUtteranceSegmenter segmenter_;
    UtteranceAnalyzer analyzer_;
    Transcriber transcribe_;
    std::vector<std::string> ready_json_;
    bool had_asr_error_ = false;
};

/// @}

}  // namespace edge_audio
