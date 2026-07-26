#include "robot/robot_event.h"

#include <cmath>
#include <iomanip>
#include <sstream>

namespace pipeline {
namespace {

std::string json_escape(const std::string& value) {
    static constexpr char kHex[] = "0123456789abcdef";
    std::string escaped;
    escaped.reserve(value.size() + 8);
    for (unsigned char c : value) {
        switch (c) {
            case '"':  escaped += "\\\""; break;
            case '\\': escaped += "\\\\"; break;
            case '\b': escaped += "\\b"; break;
            case '\f': escaped += "\\f"; break;
            case '\n': escaped += "\\n"; break;
            case '\r': escaped += "\\r"; break;
            case '\t': escaped += "\\t"; break;
            default:
                if (c < 0x20) {
                    escaped += "\\u00";
                    escaped.push_back(kHex[(c >> 4) & 0x0f]);
                    escaped.push_back(kHex[c & 0x0f]);
                } else {
                    escaped.push_back(static_cast<char>(c));
                }
        }
    }
    return escaped;
}

void string_field(
    std::ostringstream& out,
    const char* name,
    const std::string& value,
    bool comma = true) {
    out << '"' << name << "\":\"" << json_escape(value) << '"';
    if (comma) out << ',';
}

double finite_or_zero(double value) noexcept {
    return std::isfinite(value) ? value : 0.0;
}

}  // namespace

const char* segment_reason_str(SegmentReason reason) noexcept {
    switch (reason) {
        case SegmentReason::end_silence:   return "end_silence";
        case SegmentReason::max_duration:  return "max_duration";
        case SegmentReason::end_of_stream: return "end_of_stream";
    }
    return "end_of_stream";
}

const char* robot_scene_str(SceneLabel scene) noexcept {
    switch (scene) {
        case SceneLabel::SILENCE:            return "silence";
        case SceneLabel::NOISE:              return "noise";
        case SceneLabel::MUSIC:              return "music";
        case SceneLabel::SPEECH:             return "speech";
        case SceneLabel::MIXED_SPEECH_NOISE: return "mixed";
        case SceneLabel::UNKNOWN:            return "unknown";
    }
    return "unknown";
}

std::string serialize_robot_event(const RobotEvent& event) {
    std::ostringstream out;
    out << std::setprecision(17);
    out << '{';
    string_field(out, "event", "robot_voice_command");
    out << "\"utterance_id\":" << event.utterance_id << ',';
    string_field(out, "capture_device", event.capture_device);
    string_field(out, "transport", event.transport);
    out << "\"duration_ms\":" << event.duration_ms << ',';
    string_field(
        out, "finalization_reason",
        segment_reason_str(event.finalization_reason));
    string_field(out, "scene", robot_scene_str(event.scene));
    out << "\"scene_confidence\":"
        << finite_or_zero(event.scene_confidence) << ',';
    string_field(out, "quality_policy", event.quality_policy);
    out << "\"quality_threshold\":"
        << finite_or_zero(event.quality_threshold) << ',';
    out << "\"quality_probability\":"
        << finite_or_zero(event.quality_probability) << ',';
    out << "\"rule_summary\":"
        << (event.rule_summary ? "true" : "false") << ',';
    out << "\"admitted\":" << (event.admitted ? "true" : "false") << ',';
    string_field(out, "rejection_reason", event.rejection_reason);
    out << "\"asr_ran\":" << (event.asr_ran ? "true" : "false") << ',';
    string_field(out, "asr_error", event.asr_error);
    string_field(out, "transcript", event.transcript);
    string_field(out, "intent", robot_intent_str(event.route.intent));
    out << "\"action\":";
    if (event.route.has_action) {
        out << '"' << json_escape(event.route.action) << '"';
    } else {
        out << "null";
    }
    out << ',';
    out << "\"latency\":{";
    out << "\"quality_us\":" << finite_or_zero(event.quality_us) << ',';
    out << "\"asr_ms\":" << finite_or_zero(event.asr_ms) << ',';
    out << "\"post_utterance_ms\":"
        << finite_or_zero(event.post_utterance_ms);
    out << "},";
    out << "\"stream\":{";
    out << "\"frames_processed\":" << event.frames_processed << ',';
    out << "\"invalid_frames\":" << event.invalid_frames;
    out << "}}";
    return out.str();
}

}  // namespace pipeline
