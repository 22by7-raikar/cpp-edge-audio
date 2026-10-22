/// @file
/// @brief Frame-level FFT features aggregated over a chunk.
/// @ingroup quality

#pragma once

namespace edge_audio {

/// @defgroup quality Quality admission
/// @brief Deciding, before ASR, whether audio is worth transcribing.
///
/// Every analysis window is described by 12 time-domain and spectral metrics
/// (GateMetrics). Three consumers use them:
///
/// - evaluate_chunk(): an interpretable rule gate (PASS / BORDERLINE / FAIL);
/// - classify(): a rule-based scene label (speech, noise, music, ...) that the
///   AdaptiveController uses to retune the gate and skip ASR on music/silence;
/// - QualityFeatureAggregator + predict_quality(): a 27-feature file-level
///   vector scored by a gradient-boosted tree ensemble exported from Python.
///
/// decide_quality_admission() combines the rule and learned decisions under a
/// QualityPolicy, and UtteranceAnalyzer runs the whole path on one
/// utterance.
/// @{

/// @brief Analysis framing and band edges for extract_chunk_features().
struct FrameConfig {
    int frame_size = 512;              ///< FFT size (power of two); 32 ms at 16 kHz.
    int hop_size = 256;                ///< Frame hop in samples (50% overlap).
    float rolloff_percentile = 0.85f;  ///< Energy fraction defining spectral rolloff.
    float active_rms_thresh = 0.005f;  ///< Frame RMS above which a frame is "active".
    float band_low_max_hz = 500.0f;    ///< Upper edge of the low band.
    float band_mid_max_hz = 4000.0f;   ///< Upper edge of the mid (voice) band.
};

/// @brief Spectral statistics over all frames of a chunk.
///
/// Band energies are normalised so that low + mid + high ≈ 1.
struct ChunkFeatures {
    double flatness_mean = 0.0;      ///< Mean spectral flatness in [0, 1]; ~1 for white noise.
    double flatness_max = 0.0;       ///< Maximum spectral flatness.
    double centroid_mean = 0.0;      ///< Mean spectral centroid, Hz.
    double centroid_median = 0.0;    ///< Median spectral centroid, Hz.
    double rolloff_mean = 0.0;       ///< Mean rolloff frequency, Hz.
    double flux_mean = 0.0;          ///< Mean frame-to-frame spectral flux.
    double flux_max = 0.0;           ///< Maximum spectral flux.
    double band_low_mean = 0.0;      ///< Mean energy fraction below `band_low_max_hz`.
    double band_mid_mean = 0.0;      ///< Mean energy fraction in the voice band.
    double band_high_mean = 0.0;     ///< Mean energy fraction above `band_mid_max_hz`.
    int n_frames = 0;                ///< Frames analysed.
    double active_frame_frac = 0.0;  ///< Fraction of frames above `active_rms_thresh`.
};

/// @brief Compute ChunkFeatures with a Hann-windowed radix-2 FFT.
///
/// Only complete frames are analysed; a trailing partial frame is ignored, so
/// an input shorter than `frame_size` (or an invalid configuration) yields
/// all-zero features with `n_frames == 0`.
///
/// @param samples     Mono PCM.
/// @param n_samples   Number of samples.
/// @param sample_rate Sample rate in Hz (band edges are converted to FFT bins).
/// @param cfg         Framing parameters.
ChunkFeatures extract_chunk_features(const float* samples, int n_samples, int sample_rate,
                                     const FrameConfig& cfg);

/// @}

}  // namespace edge_audio
