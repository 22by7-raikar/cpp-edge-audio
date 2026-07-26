#pragma once

#include <string>
#include <string_view>

namespace pipeline {

enum class RobotIntent {
    LOOK_LEFT,
    LOOK_RIGHT,
    STOP,
    APPROACH,
    CANCEL,
    UNKNOWN,
};

struct CommandRoute {
    RobotIntent intent = RobotIntent::UNKNOWN;
    std::string normalized_transcript;
    std::string action;
    bool has_action = false;
};

std::string normalize_command(std::string_view transcript);
CommandRoute route_command(std::string_view transcript);
const char* robot_intent_str(RobotIntent intent) noexcept;

}  // namespace pipeline
