#include "edge_audio/command/command_router.h"

#include <cctype>
#include <utility>

namespace edge_audio {
namespace {

bool is_terminal_punctuation(char c) noexcept {
    return c == '.' || c == ',' || c == '!' || c == '?' || c == ';' || c == ':';
}

char ascii_lower(unsigned char byte) noexcept {
    if (byte >= 'A' && byte <= 'Z') {
        return static_cast<char>(byte - 'A' + 'a');
    }
    return static_cast<char>(byte);
}

CommandRoute matched(CommandIntent intent, std::string normalized, const char* action) {
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

    while (!normalized.empty() && is_terminal_punctuation(normalized.back())) {
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
        return matched(CommandIntent::GO_FORWARD, std::move(normalized), "go_forward");
    }
    if (normalized == "go backward") {
        return matched(CommandIntent::GO_BACKWARD, std::move(normalized), "go_backward");
    }
    if (normalized == "turn left") {
        return matched(CommandIntent::TURN_LEFT, std::move(normalized), "turn_left");
    }
    if (normalized == "turn right") {
        return matched(CommandIntent::TURN_RIGHT, std::move(normalized), "turn_right");
    }
    if (normalized == "look left") {
        return matched(CommandIntent::LOOK_LEFT, std::move(normalized), "look_left");
    }
    if (normalized == "look right") {
        return matched(CommandIntent::LOOK_RIGHT, std::move(normalized), "look_right");
    }
    if (normalized == "stop") {
        return matched(CommandIntent::STOP, std::move(normalized), "stop");
    }
    if (normalized == "come here") {
        return matched(CommandIntent::APPROACH, std::move(normalized), "approach");
    }
    if (normalized == "cancel") {
        return matched(CommandIntent::CANCEL, std::move(normalized), "cancel");
    }

    CommandRoute route;
    route.normalized_transcript = std::move(normalized);
    return route;
}

const char* command_intent_str(CommandIntent intent) noexcept {
    switch (intent) {
        case CommandIntent::GO_FORWARD:
            return "GO_FORWARD";
        case CommandIntent::GO_BACKWARD:
            return "GO_BACKWARD";
        case CommandIntent::TURN_LEFT:
            return "TURN_LEFT";
        case CommandIntent::TURN_RIGHT:
            return "TURN_RIGHT";
        case CommandIntent::LOOK_LEFT:
            return "LOOK_LEFT";
        case CommandIntent::LOOK_RIGHT:
            return "LOOK_RIGHT";
        case CommandIntent::STOP:
            return "STOP";
        case CommandIntent::APPROACH:
            return "APPROACH";
        case CommandIntent::CANCEL:
            return "CANCEL";
        case CommandIntent::UNKNOWN:
            return "UNKNOWN";
    }
    return "UNKNOWN";
}

}  // namespace edge_audio
