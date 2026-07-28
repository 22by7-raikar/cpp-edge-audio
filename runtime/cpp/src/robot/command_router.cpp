#include "robot/command_router.h"

#include <cctype>
#include <utility>

namespace pipeline {
namespace {

bool is_terminal_punctuation(char c) noexcept {
    return c == '.' || c == ',' || c == '!' || c == '?' || c == ';' ||
           c == ':';
}

char ascii_lower(unsigned char byte) noexcept {
    if (byte >= 'A' && byte <= 'Z') {
        return static_cast<char>(byte - 'A' + 'a');
    }
    return static_cast<char>(byte);
}

CommandRoute matched(
    RobotIntent intent,
    std::string normalized,
    const char* action) {
    CommandRoute route;
    route.intent = intent;
    route.normalized_transcript = std::move(normalized);
    route.action = action;
    route.has_action = true;
    return route;
}

}  // namespace

std::string normalize_command(std::string_view transcript) {
    std::string normalized;
    normalized.reserve(transcript.size());

    bool have_text = false;
    bool pending_space = false;
    for (unsigned char byte : transcript) {
        if (std::isspace(byte)) {
            if (have_text) pending_space = true;
            continue;
        }
        if (pending_space) {
            normalized.push_back(' ');
            pending_space = false;
        }
        normalized.push_back(ascii_lower(byte));
        have_text = true;
    }

    while (!normalized.empty() &&
           is_terminal_punctuation(normalized.back())) {
        normalized.pop_back();
    }
    while (!normalized.empty() && normalized.back() == ' ') {
        normalized.pop_back();
    }
    return normalized;
}

CommandRoute route_command(std::string_view transcript) {
    std::string normalized = normalize_command(transcript);
    if (normalized == "go forward") {
        return matched(
            RobotIntent::GO_FORWARD, std::move(normalized), "go_forward");
    }
    if (normalized == "go backward") {
        return matched(
            RobotIntent::GO_BACKWARD, std::move(normalized), "go_backward");
    }
    if (normalized == "turn left") {
        return matched(
            RobotIntent::TURN_LEFT, std::move(normalized), "turn_left");
    }
    if (normalized == "turn right") {
        return matched(
            RobotIntent::TURN_RIGHT, std::move(normalized), "turn_right");
    }
    if (normalized == "look left") {
        return matched(RobotIntent::LOOK_LEFT, std::move(normalized), "look_left");
    }
    if (normalized == "look right") {
        return matched(
            RobotIntent::LOOK_RIGHT, std::move(normalized), "look_right");
    }
    if (normalized == "stop") {
        return matched(RobotIntent::STOP, std::move(normalized), "stop");
    }
    if (normalized == "come here") {
        return matched(RobotIntent::APPROACH, std::move(normalized), "approach");
    }
    if (normalized == "cancel") {
        return matched(RobotIntent::CANCEL, std::move(normalized), "cancel");
    }

    CommandRoute route;
    route.normalized_transcript = std::move(normalized);
    return route;
}

const char* robot_intent_str(RobotIntent intent) noexcept {
    switch (intent) {
        case RobotIntent::GO_FORWARD: return "GO_FORWARD";
        case RobotIntent::GO_BACKWARD: return "GO_BACKWARD";
        case RobotIntent::TURN_LEFT:  return "TURN_LEFT";
        case RobotIntent::TURN_RIGHT: return "TURN_RIGHT";
        case RobotIntent::LOOK_LEFT:  return "LOOK_LEFT";
        case RobotIntent::LOOK_RIGHT: return "LOOK_RIGHT";
        case RobotIntent::STOP:       return "STOP";
        case RobotIntent::APPROACH:   return "APPROACH";
        case RobotIntent::CANCEL:     return "CANCEL";
        case RobotIntent::UNKNOWN:    return "UNKNOWN";
    }
    return "UNKNOWN";
}

}  // namespace pipeline
