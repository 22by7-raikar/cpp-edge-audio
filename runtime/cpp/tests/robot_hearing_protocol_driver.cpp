#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "robot/robot_hearing_loop.h"

namespace {

constexpr int kRate = 16000;
constexpr int kFrameSamples = 320;

std::vector<float> speech_frame() {
    std::vector<float> frame(kFrameSamples);
    for (int index = 0; index < kFrameSamples; ++index) {
        frame[static_cast<std::size_t>(index)] =
            0.2f * std::sin(
                2.0 * 3.14159265358979323846 * 1000.0 * index / kRate);
    }
    return frame;
}

void push_frames(
    pipeline::RobotHearingLoop& loop,
    const std::vector<float>& frame,
    int count) {
    for (int index = 0; index < count; ++index) {
        loop.push(frame.data(), static_cast<int>(frame.size()));
    }
}

}  // namespace

int main() {
    pipeline::RobotHearingConfig config;
    config.segmenter.sample_rate = kRate;
    config.quality_policy = pipeline::QualityPolicy::RULE;
    config.capture_device = "protocol_fixture";
    config.transport = "test_callback";

    int asr_calls = 0;
    pipeline::RobotHearingLoop loop(
        config,
        [&](const float*, int) {
            pipeline::AsrResult result;
            result.ok = true;
            result.text = asr_calls++ == 0
                ? "Look left!"
                : "please do not stop now";
            result.inference_ms = 1.25;
            return result;
        });

    const std::vector<float> silence(kFrameSamples, 0.0f);
    const std::vector<float> speech = speech_frame();

    // Silence alone is discarded and emits no event.
    push_frames(loop, silence, 40);

    // First utterance closes through the end-silence boundary.
    push_frames(loop, speech, 15);
    push_frames(loop, silence, 30);

    // Second utterance closes through idempotent EOF flushing.
    push_frames(loop, silence, 12);
    push_frames(loop, speech, 15);
    loop.flush();
    loop.flush();

    for (const auto& json : loop.pop_json_events()) {
        std::cout << json << '\n';
    }
    return asr_calls == 2 ? 0 : 1;
}
