/// @file
/// @brief The per-utterance event emitted by the streaming pipeline, and its JSON form.
/// @ingroup command

#pragma once

#include <cstdint>
#include <string>

#include "edge_audio/command/command_router.h"
#include "edge_audio/quality/quality_model.h"
#include "edge_audio/quality/scene_classifier.h"
#include "edge_audio/segmentation/utterance_segmenter.h"

namespace edge_audio {

/// @addtogroup command
/// @{

/// @brief One completed utterance: how it was segmented, whether it was
/// admitted, what ASR heard, and which command (if any) it maps to.
///
/// Every utterance produces exactly one event, including rejected ones, so the
/// consumer sees every decision. Field names match the JSON keys written by
/// serialize_command_event(); see docs/architecture.md for the schema.
struct VoiceCommandEvent {
    uint32_t utterance_id = 0;               ///< From the segmenter.
    std::string capture_device = "unknown";  ///< Caller-supplied source label.
    std::string transport = "stdin_pcm";     ///< Caller-supplied transport label.
    int duration_ms = 0;                     ///< Utterance length.
    SegmentReason finalization_reason = SegmentReason::end_silence;  ///< Why it closed.
    SceneLabel scene = SceneLabel::UNKNOWN;                          ///< Highest-confidence scene.
    double scene_confidence = 0.0;                                   ///< Heuristic confidence.
    std::string quality_policy = "rule";                             ///< Policy name.
    double quality_threshold = kQualityDefaultThreshold;             ///< Learned threshold.
    double quality_probability = 0.0;  ///< Learned probability (0 under RULE).
    bool rule_summary = false;         ///< Rule decision.
    bool admitted = false;             ///< Final verdict.
    std::string rejection_reason;      ///< Empty when admitted.
    bool asr_ran = false;              ///< Whether ASR was invoked.
    std::string asr_error;             ///< ASR failure, if any.
    std::string transcript;            ///< ASR text (empty if not run or failed).
    CommandRoute route;                ///< Routing result.
    double quality_us = 0.0;           ///< Quality analysis time.
    double asr_ms = 0.0;               ///< ASR decode time.
    double post_utterance_ms = 0.0;    ///< End-of-utterance to event, total.
    uint64_t frames_processed = 0;     ///< Stream frames up to this utterance.
};

/// @brief Lower-case name of a SegmentReason, e.g. `end_silence`.
const char* segment_reason_str(SegmentReason reason) noexcept;

/// @brief Lower-case scene name used in events (`mixed` for MIXED_SPEECH_NOISE).
const char* scene_json_str(SceneLabel scene) noexcept;

/// @brief Serialise as a single-line JSON object (no trailing newline).
///
/// Strings are escaped per RFC 8259; non-finite numbers are written as 0 so the
/// output is always valid JSON.
std::string serialize_command_event(const VoiceCommandEvent& event);

/// @}

}  // namespace edge_audio
