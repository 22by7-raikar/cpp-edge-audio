/// @file
/// @brief Exact-match mapping from transcripts to command intents.
/// @ingroup command

#pragma once

#include <string>
#include <string_view>

namespace edge_audio {

/// @defgroup command Voice commands
/// @brief Turning admitted utterances into structured command events.
///
/// VoiceCommandLoop drives the streaming path: segmenter → UtteranceAnalyzer →
/// transcribe_if_admitted() → route_command() → one VoiceCommandEvent (JSON)
/// per utterance.
/// @{

/// @brief Recognised command, or UNKNOWN.
enum class CommandIntent {
    GO_FORWARD,   ///< "go forward"
    GO_BACKWARD,  ///< "go backward"
    TURN_LEFT,    ///< "turn left"
    TURN_RIGHT,   ///< "turn right"
    LOOK_LEFT,    ///< "look left"
    LOOK_RIGHT,   ///< "look right"
    STOP,         ///< "stop"
    APPROACH,     ///< "come here"
    CANCEL,       ///< "cancel"
    UNKNOWN,      ///< Anything else.
};

/// @brief Routing result.
struct CommandRoute {
    CommandIntent intent = CommandIntent::UNKNOWN;  ///< Matched intent.
    std::string normalized_transcript;              ///< Text that was matched.
    std::string action;                             ///< snake_case action name; empty if UNKNOWN.
    bool has_action = false;                        ///< False for UNKNOWN.
};

/// @brief Lower-case ASCII, collapse whitespace, trim, and strip trailing
/// sentence punctuation (period, comma, exclamation and question marks,
/// semicolon, colon): the variation whisper adds to a short command.
std::string normalize_command(std::string_view transcript);

/// @brief Map a transcript to an intent by exact match after normalize_command().
///
/// Deliberately conservative: anything other than one of the nine phrases is
/// UNKNOWN with no action, so a mis-transcription cannot become a different
/// command.
CommandRoute route_command(std::string_view transcript);

/// @brief Upper-case intent name, e.g. `TURN_LEFT`.
const char* command_intent_str(CommandIntent intent) noexcept;

/// @}

}  // namespace edge_audio
