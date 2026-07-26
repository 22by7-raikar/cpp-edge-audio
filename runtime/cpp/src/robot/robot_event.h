#pragma once

#include <cstdint>
#include <string>

#include "chunker/streaming_utterance_segmenter.h"
#include "gate/quality_model.h"
#include "robot/command_router.h"
#include "scene/scene.h"

namespace pipeline {

struct RobotEvent {
    uint32_t utterance_id = 0;
    std::string capture_device = "unknown";
    std::string transport = "stdin_pcm";
    int duration_ms = 0;
    SegmentReason finalization_reason = SegmentReason::end_silence;
    SceneLabel scene = SceneLabel::UNKNOWN;
    double scene_confidence = 0.0;
    std::string quality_policy = "rule";
    double quality_threshold = kQualityDefaultThreshold;
    double quality_probability = 0.0;
    bool rule_summary = false;
    bool admitted = false;
    std::string rejection_reason;
    bool asr_ran = false;
    std::string asr_error;
    std::string transcript;
    CommandRoute route;
    double quality_us = 0.0;
    double asr_ms = 0.0;
    double post_utterance_ms = 0.0;
    uint64_t frames_processed = 0;
    uint64_t invalid_frames = 0;
};

const char* segment_reason_str(SegmentReason reason) noexcept;
const char* robot_scene_str(SceneLabel scene) noexcept;
std::string serialize_robot_event(const RobotEvent& event);

}  // namespace pipeline
