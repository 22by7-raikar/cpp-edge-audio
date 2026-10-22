/// @file
/// @brief Energy/zero-crossing voice activity detection and ASR window packing.
/// @ingroup segmentation

#pragma once

#include <vector>

namespace edge_audio {

/// @addtogroup segmentation
/// @{

/// @brief Frame classifier and region post-processing parameters.
struct VadConfig {
    int frame_ms = 20;             ///< Analysis frame length (320 samples at 16 kHz).
    int hop_ms = 10;               ///< Frame hop.
    float energy_thresh = 0.005f;  ///< Minimum frame RMS for speech.
    /// Maximum zero-crossing rate for speech, in crossings/s. Voiced speech is
    /// roughly 60–700; white noise roughly 8000.
    float zcr_max = 3000.0f;
    int hangover_frames = 8;   ///< Frames speech is held after the last active frame.
    int min_speech_ms = 200;   ///< Regions shorter than this are dropped.
    int min_silence_ms = 100;  ///< Gaps shorter than this are merged.
};

/// @brief A detected speech region.
struct VadSegment {
    double start_sec = 0.0;     ///< Region start, seconds.
    double end_sec = 0.0;       ///< Region end, seconds.
    float speech_ratio = 0.0f;  ///< Fraction of frames in the region classified as speech.
    int frame_count = 0;        ///< Number of hop frames spanned.

    /// @brief Region length in seconds.
    double duration_sec() const { return end_sec - start_sec; }
};

/// @brief Classify one frame: speech iff RMS >= `energy_thresh` and ZCR < `zcr_max`.
///
/// This is the per-frame decision shared by run_vad() and
/// StreamingUtteranceSegmenter::push().
bool vad_frame_is_speech(const float* frame, int frame_samples, int sample_rate,
                         const VadConfig& cfg = VadConfig{});

/// @brief Detect speech regions in a complete buffer.
///
/// Classifies frames with vad_frame_is_speech(), applies hangover, merges gaps
/// shorter than `min_silence_ms`, and drops regions shorter than `min_speech_ms`.
///
/// @return Regions in chronological order; empty if no speech is found.
std::vector<VadSegment> run_vad(const float* samples, int n_samples, int sample_rate,
                                const VadConfig& cfg = VadConfig{});

/// @brief Parameters for pack_vad_segments().
struct VadPackConfig {
    int pre_pad_ms = 200;      ///< Context added before each region.
    int post_pad_ms = 300;     ///< Context added after each region.
    int merge_gap_ms = 600;    ///< Merge padded windows separated by at most this gap.
    int min_window_ms = 1500;  ///< Short windows are extended to this length.
    int max_window_ms = 7000;  ///< Merging never produces a window longer than this.
};

/// @brief Pack VAD regions into fewer, wider ASR windows.
///
/// Each ASR call has a fixed overhead (notably on GPU), so many short regions
/// are slower to transcribe than a few padded windows covering them. Windows
/// are padded, clipped to [0, @p audio_dur_sec], greedily merged within
/// `merge_gap_ms` and `max_window_ms`, then extended to `min_window_ms`.
std::vector<VadSegment> pack_vad_segments(const std::vector<VadSegment>& segs, double audio_dur_sec,
                                          const VadPackConfig& cfg = VadPackConfig{});

/// @}

}  // namespace edge_audio
