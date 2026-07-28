#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "robot/command_router.h"
#include "robot/robot_event.h"
#include "robot/robot_hearing_loop.h"

namespace {

int passed = 0;
int failed = 0;

#define CHECK(condition)                                                   \
    do {                                                                   \
        if (!(condition)) {                                                \
            std::cerr << "FAIL: " << __FILE__ << ":" << __LINE__         \
                      << "  " << #condition << "\n";                     \
            return false;                                                  \
        }                                                                  \
    } while (0)

bool run(const char* name, bool (*test)()) {
    const bool ok = test();
    std::cout << "  " << (ok ? "PASS" : "FAIL") << "  " << name << "\n";
    ok ? ++passed : ++failed;
    return ok;
}

pipeline::Utterance utterance(
    uint32_t id,
    bool speech,
    pipeline::SegmentReason reason = pipeline::SegmentReason::end_silence) {
    pipeline::Utterance output;
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
                0.2f * std::sin(
                    2.0 * 3.14159265358979323846 * 1000.0 * index / 16000.0);
        }
    }
    return output;
}

bool test_command_normalization_and_routes() {
    CHECK(pipeline::normalize_command(" \tTURN   RIGHT: \r\n") == "turn right");

    struct Case {
        const char* text;
        pipeline::RobotIntent intent;
        const char* action;
    };
    const Case cases[] = {
        {"stop", pipeline::RobotIntent::STOP, "stop"},
        {"GO FORWARD", pipeline::RobotIntent::GO_FORWARD, "go_forward"},
        {"Go   Backward!", pipeline::RobotIntent::GO_BACKWARD, "go_backward"},
        {" turn   left, ", pipeline::RobotIntent::TURN_LEFT, "turn_left"},
        {" turn right ", pipeline::RobotIntent::TURN_RIGHT, "turn_right"},
        {"turn right?", pipeline::RobotIntent::TURN_RIGHT, "turn_right"},
        {"Go forward.", pipeline::RobotIntent::GO_FORWARD, "go_forward"},
        {"Stop.", pipeline::RobotIntent::STOP, "stop"},
        {"stop;", pipeline::RobotIntent::STOP, "stop"},
        {"stop:", pipeline::RobotIntent::STOP, "stop"},
        {"look left.", pipeline::RobotIntent::LOOK_LEFT, "look_left"},
        {"LOOK RIGHT?", pipeline::RobotIntent::LOOK_RIGHT, "look_right"},
        {" stop! ", pipeline::RobotIntent::STOP, "stop"},
        {"come here,", pipeline::RobotIntent::APPROACH, "approach"},
        {"cancel", pipeline::RobotIntent::CANCEL, "cancel"},
    };
    for (const auto& item : cases) {
        const auto route = pipeline::route_command(item.text);
        CHECK(route.intent == item.intent);
        CHECK(route.has_action);
        CHECK(route.action == item.action);
    }

    const char* unknown_cases[] = {
        "Go back load.",
        "You're on left.",
        "go",
        "forward",
        "turn",
        "do not stop",
        "stop now",
        "",
        " ;:?!,.",
    };
    for (const char* text : unknown_cases) {
        const auto route = pipeline::route_command(text);
        CHECK(route.intent == pipeline::RobotIntent::UNKNOWN);
        CHECK(!route.has_action);
        CHECK(route.action.empty());
    }
    return true;
}

bool test_json_escaping_and_null_action() {
    pipeline::RobotEvent event;
    event.utterance_id = 7;
    event.capture_device = "fixture\"device";
    event.transport = "pipe\ntransport";
    event.transcript = "unknown\tcommand\\value";
    event.route = pipeline::route_command(event.transcript);
    const std::string json = pipeline::serialize_robot_event(event);

    CHECK(json.find("fixture\\\"device") != std::string::npos);
    CHECK(json.find("pipe\\ntransport") != std::string::npos);
    CHECK(json.find("unknown\\tcommand\\\\value") != std::string::npos);
    CHECK(json.find("\"intent\":\"UNKNOWN\"") != std::string::npos);
    CHECK(json.find("\"action\":null") != std::string::npos);
    CHECK(json.find('\n') == std::string::npos);
    return true;
}

bool test_learned_rejection_emits_event_and_skips_asr() {
    pipeline::RobotHearingConfig config;
    config.segmenter.sample_rate = 16000;
    config.quality_policy = pipeline::QualityPolicy::LEARNED;
    int calls = 0;
    pipeline::RobotHearingLoop loop(
        config,
        [&](const float*, int) {
            ++calls;
            return pipeline::AsrResult{};
        });

    const std::string json =
        loop.process_completed_utterance(utterance(1, false));
    CHECK(calls == 0);
    CHECK(json.find("\"admitted\":false") != std::string::npos);
    CHECK(json.find("\"asr_ran\":false") != std::string::npos);
    CHECK(json.find("\"rejection_reason\":\"learned_below_threshold\"") !=
          std::string::npos);
    CHECK(json.find("\"action\":null") != std::string::npos);
    return true;
}

bool test_admission_calls_asr_once_and_routes() {
    pipeline::RobotHearingConfig config;
    config.segmenter.sample_rate = 16000;
    config.quality_policy = pipeline::QualityPolicy::RULE;
    int calls = 0;
    bool pointer_ok = false;
    bool count_ok = false;
    pipeline::RobotHearingLoop loop(
        config,
        [&](const float* samples, int count) {
            ++calls;
            pointer_ok = samples != nullptr;
            count_ok = count == 16000;
            pipeline::AsrResult result;
            result.ok = true;
            result.text = "Go forward.";
            result.inference_ms = 12.5;
            return result;
        });

    const std::string json =
        loop.process_completed_utterance(utterance(1, true));
    CHECK(calls == 1);
    CHECK(pointer_ok);
    CHECK(count_ok);
    CHECK(json.find("\"admitted\":true") != std::string::npos);
    CHECK(json.find("\"asr_ran\":true") != std::string::npos);
    CHECK(json.find("\"intent\":\"GO_FORWARD\"") != std::string::npos);
    CHECK(json.find("\"action\":\"go_forward\"") != std::string::npos);
    return true;
}

bool test_admitted_unknown_transcript_has_null_action() {
    pipeline::RobotHearingConfig config;
    config.segmenter.sample_rate = 16000;
    config.quality_policy = pipeline::QualityPolicy::RULE;
    pipeline::RobotHearingLoop loop(
        config,
        [](const float*, int) {
            pipeline::AsrResult result;
            result.ok = true;
            result.text = "Go back load.";
            return result;
        });

    const std::string json =
        loop.process_completed_utterance(utterance(1, true));
    CHECK(json.find("\"asr_ran\":true") != std::string::npos);
    CHECK(json.find("\"transcript\":\"Go back load.\"") !=
          std::string::npos);
    CHECK(json.find("\"intent\":\"UNKNOWN\"") != std::string::npos);
    CHECK(json.find("\"action\":null") != std::string::npos);
    return true;
}

bool test_asr_failure_remains_valid_event() {
    pipeline::RobotHearingConfig config;
    config.segmenter.sample_rate = 16000;
    pipeline::RobotHearingLoop loop(
        config,
        [](const float*, int) {
            pipeline::AsrResult result;
            result.error = "decoder \"failure\"\n";
            return result;
        });

    const std::string json =
        loop.process_completed_utterance(utterance(1, true));
    CHECK(loop.had_asr_error());
    CHECK(json.find("\"asr_ran\":true") != std::string::npos);
    CHECK(json.find("decoder \\\"failure\\\"\\n") != std::string::npos);
    CHECK(json.find("\"action\":null") != std::string::npos);
    CHECK(json.find('\n') == std::string::npos);
    return true;
}

}  // namespace

int main() {
    std::cout << "Running robot-hearing loop tests...\n\n";
    run("command_normalization_routes", test_command_normalization_and_routes);
    run("json_escaping_null_action", test_json_escaping_and_null_action);
    run("learned_rejection_one_event_zero_asr",
        test_learned_rejection_emits_event_and_skips_asr);
    run("admission_exactly_one_asr", test_admission_calls_asr_once_and_routes);
    run("admitted_unknown_null_action",
        test_admitted_unknown_transcript_has_null_action);
    run("asr_failure_valid_event", test_asr_failure_remains_valid_event);
    std::cout << "\n" << passed << " passed, " << failed << " failed\n";
    return failed == 0 ? 0 : 1;
}
