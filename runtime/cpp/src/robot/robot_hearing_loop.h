#pragma once

#include <string>
#include <vector>

#include "chunker/streaming_utterance_segmenter.h"
#include "gate/quality_policy.h"
#include "pipeline/completed_utterance_analyzer.h"

namespace pipeline {

struct RobotHearingConfig {
    SegmenterConfig segmenter;
    QualityPolicy quality_policy = QualityPolicy::RULE;
    double quality_threshold = kQualityDefaultThreshold;
    std::string capture_device = "unknown";
    std::string transport = "stdin_pcm";
};

// Stateful streaming loop with an injected ASR callback. The callback seam
// keeps deterministic tests independent of a real Whisper model.
class RobotHearingLoop {
public:
    RobotHearingLoop(RobotHearingConfig config, FileAsrCallback asr_callback);

    void push(const float* samples, int sample_count);
    void flush();

    // Returns complete newline-free JSON records and clears the ready queue.
    std::vector<std::string> pop_json_events();

    // Exposed for deterministic completed-utterance tests. Streaming paths
    // reach the same implementation after the segmenter emits an utterance.
    std::string process_completed_utterance(Utterance utterance);

    const SegmenterTelemetry& telemetry() const noexcept {
        return segmenter_.telemetry();
    }
    bool had_asr_error() const noexcept { return had_asr_error_; }

private:
    RobotHearingConfig config_;
    StreamingUtteranceSegmenter segmenter_;
    CompletedUtteranceAnalyzer analyzer_;
    FileAsrCallback asr_callback_;
    std::vector<std::string> ready_json_;
    bool had_asr_error_ = false;

    void drain_completed();
};

}  // namespace pipeline
