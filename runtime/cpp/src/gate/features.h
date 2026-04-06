#pragma once

#include <vector>

namespace pipeline {

// -------------------------------------------------------
// Per-frame spectral features (one struct per FFT frame)
// -------------------------------------------------------
struct FrameFeatures {
    float flatness     = 0.0f;  // geometric/arithmetic mean ratio of power spectrum [0,1]
    float centroid_hz  = 0.0f;  // spectral centroid in Hz
    float rolloff_hz   = 0.0f;  // frequency below which 85% of energy is contained
    float flux         = 0.0f;  // sum of squared magnitude diff vs previous frame
    float band_low     = 0.0f;  // normalized energy in 0 - 500 Hz
    float band_mid     = 0.0f;  // normalized energy in 500 - 4000 Hz
    float band_high    = 0.0f;  // normalized energy in 4000 - Nyquist Hz
    float frame_rms    = 0.0f;  // RMS of this frame's time-domain samples
};

// -------------------------------------------------------
// Chunk-level aggregated features
// -------------------------------------------------------
struct ChunkFeatures {
    // Aggregates across all frames
    double flatness_mean   = 0.0;
    double flatness_max    = 0.0;

    double centroid_mean   = 0.0;  // Hz
    double centroid_median = 0.0;  // Hz

    double rolloff_mean    = 0.0;  // Hz

    double flux_mean       = 0.0;
    double flux_max        = 0.0;

    double band_low_mean   = 0.0;
    double band_mid_mean   = 0.0;
    double band_high_mean  = 0.0;

    int    n_frames        = 0;
    // Fraction of frames where frame_rms > active_rms_thresh
    double active_frame_frac = 0.0;
};

// -------------------------------------------------------
// Configuration for the frame-level feature extractor
// -------------------------------------------------------
struct FrameConfig {
    int   frame_size       = 512;     // FFT size, must be power of 2; 32ms at 16kHz
    int   hop_size         = 256;     // frame hop in samples; 50% overlap
    float rolloff_percentile = 0.85f; // spectral rolloff energy threshold
    float active_rms_thresh  = 0.005f;// frame considered active above this RMS

    // Band boundaries in Hz — converted to bin indices at runtime
    float band_low_max_hz  = 500.0f;
    float band_mid_max_hz  = 4000.0f;
    // band_high is [band_mid_max_hz, Nyquist]
};

// -------------------------------------------------------
// Extract chunk-level features from raw mono float32 PCM.
// samples    : pointer to chunk samples
// n_samples  : number of samples
// sample_rate: must match the sample rate the chunk was produced at (16000 Hz)
// cfg        : frame/window/band configuration
// -------------------------------------------------------
ChunkFeatures extract_chunk_features(
    const float* samples,
    int          n_samples,
    int          sample_rate,
    const FrameConfig& cfg);

}  // namespace pipeline
