#pragma once

#include <vector>

namespace pipeline {

// -------------------------------------------------------
// VAD configuration
// -------------------------------------------------------
struct VadConfig {
    int   frame_ms        = 20;      // frame length in ms  (20ms @ 16kHz = 320 samples)
    int   hop_ms          = 10;      // hop between frames  (10ms @ 16kHz = 160 samples)
    float energy_thresh   = 0.005f;  // RMS below this threshold -> non-speech
    // ZCR threshold in crossings/sec.
    // Voiced speech: ~60-700 crossings/sec.  White noise: ~8000 crossings/sec.
    float zcr_max         = 3000.0f;
    int   hangover_frames = 8;       // hold speech for this many frames after last active frame (~80ms)
    int   min_speech_ms   = 200;     // discard segments shorter than this
    int   min_silence_ms  = 100;     // merge adjacent segments with silence gap shorter than this
};

// -------------------------------------------------------
// A single detected speech segment
// -------------------------------------------------------
struct VadSegment {
    double start_sec    = 0.0;
    double end_sec      = 0.0;
    float  speech_ratio = 0.0f;  // fraction of frames in this segment with raw speech activity
    int    frame_count  = 0;     // number of VAD hop-frames spanning this segment

    double duration_sec() const { return end_sec - start_sec; }
};

// -------------------------------------------------------
// Run frame-level DSP VAD on mono float32 PCM.
//
// Algorithm:
//   1. Classify frames by RMS energy and ZCR.
//   2. Apply hangover: hold speech state for hangover_frames after last active frame.
//   3. Extract contiguous speech regions.
//   4. Merge regions whose silence gap is < min_silence_ms.
//   5. Discard regions shorter than min_speech_ms.
//
// Returns speech segments in chronological order.
// Returns empty vector if no speech is detected.
// -------------------------------------------------------
std::vector<VadSegment> run_vad(
    const float*     samples,
    int              n_samples,
    int              sample_rate,
    const VadConfig& cfg = VadConfig{});

// -------------------------------------------------------
// VAD segment packing
//
// Reduces CUDA per-call overhead by merging nearby VAD segments
// into wider ASR windows with context padding.
// -------------------------------------------------------
struct VadPackConfig {
    int pre_pad_ms    = 200;   // silence context prepended before each speech region
    int post_pad_ms   = 300;   // silence context appended after each speech region
    int merge_gap_ms  = 600;   // merge consecutive windows when gap <= this after padding
    int min_window_ms = 1500;  // extend windows shorter than this (reduces micro-call overhead)
    int max_window_ms = 7000;  // hard cap per window; start new window if exceeded
};

// Pack raw VAD segments into wider ASR windows.
//
// Steps:
//   1. Add pre/post padding to each segment, clipped to [0, audio_dur_sec].
//   2. Greedily merge adjacent padded windows if their gap <= merge_gap_ms
//      and the merged window would not exceed max_window_ms.
//   3. Extend short windows to min_window_ms.
//
// speech_ratio in output reflects raw speech time / window duration.
// frame_count is approximate (window_ms / hop_ms).
std::vector<VadSegment> pack_vad_segments(
    const std::vector<VadSegment>& segs,
    double              audio_dur_sec,
    const VadPackConfig& cfg = VadPackConfig{});

}  // namespace pipeline
