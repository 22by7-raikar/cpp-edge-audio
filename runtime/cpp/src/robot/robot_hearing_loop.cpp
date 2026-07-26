#include "robot/robot_hearing_loop.h"

#include <chrono>
#include <cmath>
#include <stdexcept>
#include <utility>

#include "robot/command_router.h"
#include "robot/robot_event.h"

namespace pipeline {

RobotHearingLoop::RobotHearingLoop(
    RobotHearingConfig config,
    FileAsrCallback asr_callback)
    : config_(std::move(config)),
      segmenter_(config_.segmenter),
      asr_callback_(std::move(asr_callback)) {
    if (config_.quality_threshold != kQualityDefaultThreshold) {
        throw std::invalid_argument(
            "robot hearing requires the frozen quality threshold 0.3");
    }
    if (!asr_callback_) {
        throw std::invalid_argument("robot hearing ASR callback is missing");
    }
}

void RobotHearingLoop::push(const float* samples, int sample_count) {
    segmenter_.push(samples, sample_count);
    drain_completed();
}

void RobotHearingLoop::flush() {
    segmenter_.flush();
    drain_completed();
}

std::vector<std::string> RobotHearingLoop::pop_json_events() {
    std::vector<std::string> output;
    output.swap(ready_json_);
    return output;
}

std::string RobotHearingLoop::process_completed_utterance(
    Utterance utterance) {
    const auto post_started = std::chrono::steady_clock::now();

    AudioBuffer input;
    input.samples = std::move(utterance.samples);
    input.sample_rate = utterance.sample_rate;
    input.channels = 1;

    CompletedUtteranceAnalysis analysis =
        analyzer_.analyze(
            std::move(input),
            config_.quality_policy,
            config_.quality_threshold);

    const FileAsrExecution execution = transcribe_admitted_file_once(
        analysis.admission.final_should_transcribe,
        true,
        analysis.audio_16khz.samples.data(),
        analysis.audio_16khz.samples.size(),
        asr_callback_);

    RobotEvent event;
    event.utterance_id = utterance.id;
    event.capture_device = config_.capture_device;
    event.transport = config_.transport;
    event.duration_ms = static_cast<int>(std::llround(
        analysis.audio_16khz.duration_sec() * 1000.0));
    event.finalization_reason = utterance.reason;
    event.scene = analysis.scene.label;
    event.scene_confidence = analysis.scene.confidence;
    event.quality_policy = quality_policy_str(config_.quality_policy);
    event.quality_threshold = config_.quality_threshold;
    event.quality_probability = analysis.learned_evaluated
        ? analysis.quality_prediction.probability
        : 0.0;
    event.rule_summary = analysis.rule_summary;
    event.admitted = analysis.admission.final_should_transcribe;
    event.rejection_reason = analysis.admission.rejection_reason;
    event.asr_ran = execution.ran;
    event.asr_ms = execution.result.inference_ms;
    event.asr_error = execution.result.error;
    if (execution.result.ok) {
        event.transcript = execution.result.text;
        event.route = route_command(event.transcript);
    } else {
        event.route = route_command("");
    }
    event.quality_us = analysis.quality_us;
    event.frames_processed = utterance.end_stream_frame >= 0
        ? static_cast<uint64_t>(utterance.end_stream_frame + 1)
        : 0;
    event.invalid_frames = 0;

    if (execution.ran && !execution.result.ok) {
        had_asr_error_ = true;
    }

    const auto post_finished = std::chrono::steady_clock::now();
    event.post_utterance_ms =
        std::chrono::duration<double, std::milli>(
            post_finished - post_started).count();
    return serialize_robot_event(event);
}

void RobotHearingLoop::drain_completed() {
    for (auto& utterance : segmenter_.pop_utterances()) {
        ready_json_.push_back(
            process_completed_utterance(std::move(utterance)));
    }
}

}  // namespace pipeline
