#include "edge_audio/command/voice_command_loop.h"

#include <chrono>
#include <cmath>
#include <stdexcept>
#include <utility>

#include "edge_audio/command/command_event.h"
#include "edge_audio/command/command_router.h"

namespace edge_audio {

VoiceCommandLoop::VoiceCommandLoop(VoiceCommandConfig config, Transcriber transcribe)
    : config_(std::move(config)),
      segmenter_(config_.segmenter),
      transcribe_(std::move(transcribe)) {
    if (!transcribe_) {
        throw std::invalid_argument("voice command loop needs a transcriber");
    }
    if (!std::isfinite(config_.quality_threshold) || config_.quality_threshold < 0.0 ||
        config_.quality_threshold > 1.0) {
        throw std::invalid_argument("quality threshold must be in [0, 1]");
    }
}

void VoiceCommandLoop::push(const float* samples, int sample_count) {
    segmenter_.push(samples, sample_count);
    drain_completed();
}

void VoiceCommandLoop::flush() {
    segmenter_.flush();
    drain_completed();
}

std::vector<std::string> VoiceCommandLoop::pop_json_events() {
    std::vector<std::string> output;
    output.swap(ready_json_);
    return output;
}

std::string VoiceCommandLoop::process_completed_utterance(Utterance utterance) {
    const auto started = std::chrono::steady_clock::now();

    AudioBuffer input;
    input.samples = std::move(utterance.samples);
    input.sample_rate = utterance.sample_rate;

    UtteranceAnalysis analysis =
        analyzer_.analyze(std::move(input), config_.quality_policy, config_.quality_threshold);

    const AsrRun asr = transcribe_if_admitted(
        analysis.admission.final_should_transcribe, /*asr_enabled=*/true,
        analysis.audio_16khz.samples.data(), analysis.audio_16khz.samples.size(), transcribe_);
    if (asr.ran && !asr.result.ok) {
        had_asr_error_ = true;
    }

    VoiceCommandEvent event;
    event.utterance_id = utterance.id;
    event.capture_device = config_.capture_device;
    event.transport = config_.transport;
    event.duration_ms =
        static_cast<int>(std::llround(analysis.audio_16khz.duration_sec() * 1000.0));
    event.finalization_reason = utterance.reason;
    event.scene = analysis.scene.label;
    event.scene_confidence = analysis.scene.confidence;
    event.quality_policy = quality_policy_str(config_.quality_policy);
    event.quality_threshold = config_.quality_threshold;
    event.quality_probability =
        analysis.learned_evaluated ? analysis.quality_prediction.probability : 0.0;
    event.rule_summary = analysis.rule_summary;
    event.admitted = analysis.admission.final_should_transcribe;
    event.rejection_reason = analysis.admission.rejection_reason;
    event.asr_ran = asr.ran;
    event.asr_ms = asr.result.inference_ms;
    event.asr_error = asr.result.error;
    if (asr.result.ok) {
        event.transcript = asr.result.text;
    }
    event.route = route_command(event.transcript);
    event.quality_us = analysis.quality_us;
    event.frames_processed =
        utterance.end_stream_frame >= 0 ? static_cast<uint64_t>(utterance.end_stream_frame + 1) : 0;
    event.post_utterance_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started)
            .count();
    return serialize_command_event(event);
}

void VoiceCommandLoop::drain_completed() {
    for (auto& utterance : segmenter_.pop_utterances()) {
        ready_json_.push_back(process_completed_utterance(std::move(utterance)));
    }
}

}  // namespace edge_audio
