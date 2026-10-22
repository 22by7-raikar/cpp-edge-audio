// Command routing, event JSON and the VoiceCommandLoop admission/ASR contract.

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "edge_audio/command/command_event.h"
#include "edge_audio/command/command_router.h"
#include "edge_audio/command/voice_command_loop.h"

namespace {

edge_audio::Utterance utterance(
    uint32_t id, bool speech,
    edge_audio::SegmentReason reason = edge_audio::SegmentReason::end_silence) {
    edge_audio::Utterance output;
    output.id = id;
    output.sample_rate = 16000;
    output.reason = reason;
    output.total_frames = 50;
    output.speech_frames = speech ? 50 : 0;
    output.start_stream_frame = 0;
    output.end_stream_frame = 49;
    output.samples.resize(16000);
    if (speech) {
        for (int index = 0; index < 16000; ++index) {
            output.samples[static_cast<std::size_t>(index)] =
                0.2f * std::sin(2.0 * 3.14159265358979323846 * 1000.0 * index / 16000.0);
        }
    }
    return output;
}

TEST(VoiceCommandLoop, CommandNormalizationAndRoutes) {
    ASSERT_EQ(edge_audio::normalize_command(" \tTURN   RIGHT: \r\n"), "turn right");

    struct Case {
        const char* text;
        edge_audio::CommandIntent intent;
        const char* action;
    };
    const Case cases[] = {
        {"stop", edge_audio::CommandIntent::STOP, "stop"},
        {"GO FORWARD", edge_audio::CommandIntent::GO_FORWARD, "go_forward"},
        {"Go   Backward!", edge_audio::CommandIntent::GO_BACKWARD, "go_backward"},
        {" turn   left, ", edge_audio::CommandIntent::TURN_LEFT, "turn_left"},
        {" turn right ", edge_audio::CommandIntent::TURN_RIGHT, "turn_right"},
        {"turn right?", edge_audio::CommandIntent::TURN_RIGHT, "turn_right"},
        {"Go forward.", edge_audio::CommandIntent::GO_FORWARD, "go_forward"},
        {"Stop.", edge_audio::CommandIntent::STOP, "stop"},
        {"stop;", edge_audio::CommandIntent::STOP, "stop"},
        {"stop:", edge_audio::CommandIntent::STOP, "stop"},
        {"look left.", edge_audio::CommandIntent::LOOK_LEFT, "look_left"},
        {"LOOK RIGHT?", edge_audio::CommandIntent::LOOK_RIGHT, "look_right"},
        {" stop! ", edge_audio::CommandIntent::STOP, "stop"},
        {"come here,", edge_audio::CommandIntent::APPROACH, "approach"},
        {"cancel", edge_audio::CommandIntent::CANCEL, "cancel"},
    };
    for (const auto& item : cases) {
        const auto route = edge_audio::route_command(item.text);
        ASSERT_EQ(route.intent, item.intent);
        ASSERT_TRUE(route.has_action);
        ASSERT_EQ(route.action, item.action);
    }

    const char* unknown_cases[] = {
        "Go back load.", "You're on left.", "go", "forward", "turn",
        "do not stop",   "stop now",        "",   " ;:?!,.",
    };
    for (const char* text : unknown_cases) {
        const auto route = edge_audio::route_command(text);
        ASSERT_EQ(route.intent, edge_audio::CommandIntent::UNKNOWN);
        ASSERT_FALSE(route.has_action);
        ASSERT_TRUE(route.action.empty());
    }
}

TEST(VoiceCommandLoop, JsonEscapingAndNullAction) {
    edge_audio::VoiceCommandEvent event;
    event.utterance_id = 7;
    event.capture_device = "fixture\"device";
    event.transport = "pipe\ntransport";
    event.transcript = "unknown\tcommand\\value";
    event.route = edge_audio::route_command(event.transcript);
    const std::string json = edge_audio::serialize_command_event(event);

    ASSERT_NE(json.find("fixture\\\"device"), std::string::npos);
    ASSERT_NE(json.find("pipe\\ntransport"), std::string::npos);
    ASSERT_NE(json.find("unknown\\tcommand\\\\value"), std::string::npos);
    ASSERT_NE(json.find("\"intent\":\"UNKNOWN\""), std::string::npos);
    ASSERT_NE(json.find("\"action\":null"), std::string::npos);
    ASSERT_EQ(json.find('\n'), std::string::npos);
}

TEST(VoiceCommandLoop, LearnedRejectionEmitsEventAndSkipsAsr) {
    edge_audio::VoiceCommandConfig config;
    config.segmenter.sample_rate = 16000;
    config.quality_policy = edge_audio::QualityPolicy::LEARNED;
    int calls = 0;
    edge_audio::VoiceCommandLoop loop(config, [&](const float*, int) {
        ++calls;
        return edge_audio::AsrResult{};
    });

    const std::string json = loop.process_completed_utterance(utterance(1, false));
    ASSERT_EQ(calls, 0);
    ASSERT_NE(json.find("\"admitted\":false"), std::string::npos);
    ASSERT_NE(json.find("\"asr_ran\":false"), std::string::npos);
    ASSERT_NE(json.find("\"rejection_reason\":\"learned_below_threshold\""), std::string::npos);
    ASSERT_NE(json.find("\"action\":null"), std::string::npos);
}

TEST(VoiceCommandLoop, AdmissionCallsAsrOnceAndRoutes) {
    edge_audio::VoiceCommandConfig config;
    config.segmenter.sample_rate = 16000;
    config.quality_policy = edge_audio::QualityPolicy::RULE;
    int calls = 0;
    bool pointer_ok = false;
    bool count_ok = false;
    edge_audio::VoiceCommandLoop loop(config, [&](const float* samples, int count) {
        ++calls;
        pointer_ok = samples != nullptr;
        count_ok = count == 16000;
        edge_audio::AsrResult result;
        result.ok = true;
        result.text = "Go forward.";
        result.inference_ms = 12.5;
        return result;
    });

    const std::string json = loop.process_completed_utterance(utterance(1, true));
    ASSERT_EQ(calls, 1);
    ASSERT_TRUE(pointer_ok);
    ASSERT_TRUE(count_ok);
    ASSERT_NE(json.find("\"admitted\":true"), std::string::npos);
    ASSERT_NE(json.find("\"asr_ran\":true"), std::string::npos);
    ASSERT_NE(json.find("\"intent\":\"GO_FORWARD\""), std::string::npos);
    ASSERT_NE(json.find("\"action\":\"go_forward\""), std::string::npos);
}

TEST(VoiceCommandLoop, AdmittedUnknownTranscriptHasNullAction) {
    edge_audio::VoiceCommandConfig config;
    config.segmenter.sample_rate = 16000;
    config.quality_policy = edge_audio::QualityPolicy::RULE;
    edge_audio::VoiceCommandLoop loop(config, [](const float*, int) {
        edge_audio::AsrResult result;
        result.ok = true;
        result.text = "Go back load.";
        return result;
    });

    const std::string json = loop.process_completed_utterance(utterance(1, true));
    ASSERT_NE(json.find("\"asr_ran\":true"), std::string::npos);
    ASSERT_NE(json.find("\"transcript\":\"Go back load.\""), std::string::npos);
    ASSERT_NE(json.find("\"intent\":\"UNKNOWN\""), std::string::npos);
    ASSERT_NE(json.find("\"action\":null"), std::string::npos);
}

TEST(VoiceCommandLoop, AsrFailureRemainsValidEvent) {
    edge_audio::VoiceCommandConfig config;
    config.segmenter.sample_rate = 16000;
    edge_audio::VoiceCommandLoop loop(config, [](const float*, int) {
        edge_audio::AsrResult result;
        result.error = "decoder \"failure\"\n";
        return result;
    });

    const std::string json = loop.process_completed_utterance(utterance(1, true));
    ASSERT_TRUE(loop.had_asr_error());
    ASSERT_NE(json.find("\"asr_ran\":true"), std::string::npos);
    ASSERT_NE(json.find("decoder \\\"failure\\\"\\n"), std::string::npos);
    ASSERT_NE(json.find("\"action\":null"), std::string::npos);
    ASSERT_EQ(json.find('\n'), std::string::npos);
}

}  // namespace
