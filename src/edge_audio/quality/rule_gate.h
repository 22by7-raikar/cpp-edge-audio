/// @file
/// @brief Interpretable rule gate over time-domain and spectral chunk metrics.
/// @ingroup quality

#pragma once

#include <string>

#include "edge_audio/quality/spectral_features.h"
#include "edge_audio/segmentation/fixed_chunker.h"

namespace edge_audio {

/// @addtogroup quality
/// @{

/// @brief Rule-gate verdict for one chunk.
enum class GateDecision {
    PASS,        ///< Speech-like; transcribe.
    FAIL,        ///< Not worth transcribing.
    BORDERLINE,  ///< Weak or noisy speech; admitted by the rule policy.
};

/// @brief Stable upper-case name of a GateDecision (used in logs and features).
const char* gate_decision_str(GateDecision d);

/// @brief The 12 per-chunk metrics shared by the gate, the scene classifier and
/// the learned model's feature vector.
struct GateMetrics {
    double duration_sec = 0.0;    ///< Chunk length (not a model feature).
    double rms = 0.0;             ///< Root-mean-square amplitude.
    double silence_ratio = 0.0;   ///< Fraction of samples below GateConfig::silence_thresh.
    double clipping_ratio = 0.0;  ///< Fraction of samples at or above GateConfig::clipping_thresh.
    double zcr = 0.0;             ///< Zero crossings per second.
    double spectral_flatness = 0.0;  ///< Mean spectral flatness; near 1 is noise-like.
    double spectral_centroid = 0.0;  ///< Mean spectral centroid, Hz.
    double spectral_rolloff = 0.0;   ///< Mean 85% rolloff, Hz.
    double spectral_flux = 0.0;      ///< Mean spectral flux.
    double band_energy_low = 0.0;    ///< Energy fraction 0–500 Hz.
    double band_energy_mid = 0.0;    ///< Energy fraction 500–4000 Hz.
    double band_energy_high = 0.0;   ///< Energy fraction above 4000 Hz.
    double active_frame_frac = 0.0;  ///< Fraction of FFT frames above the activity RMS.
};

/// @brief Gate verdict with its reason and the metrics it was based on.
///
/// `reason` is a stable identifier: `ok`, `rms_too_low`, `high_silence_ratio`,
/// `high_clipping_ratio`, `low_active_frame_fraction`, `stationary_noise_like`,
/// `weak_mid_band_speech_presence`, `excessive_high_band_energy`,
/// `borderline_low_energy` or `borderline_noisy_speech`.
struct GateResult {
    GateDecision decision = GateDecision::FAIL;  ///< Verdict.
    std::string reason;                          ///< Stable reason identifier.
    GateMetrics metrics;                         ///< Always populated, whatever the verdict.
};

/// @brief Gate thresholds. Defaults prefer BORDERLINE over a false FAIL.
///
/// @warning The learned model's features include gate decisions computed with
/// these defaults; changing them changes the model's inputs.
struct GateConfig {
    double rms_min = 0.003;                ///< Below: BORDERLINE (`borderline_low_energy`).
    double rms_borderline_min = 0.001;     ///< Below: FAIL (`rms_too_low`).
    double max_silence_ratio = 0.90;       ///< Above: FAIL.
    double max_clipping_ratio = 0.05;      ///< Above: FAIL.
    float silence_thresh = 0.005f;         ///< |sample| below this counts as silent.
    float clipping_thresh = 0.99f;         ///< |sample| at or above this counts as clipped.
    double zcr_max_noise = 400.0;          ///< ZCR that, with warn-level flatness, means noise.
    double min_active_frame_frac = 0.10;   ///< Below: FAIL.
    double spectral_flatness_max = 0.90;   ///< Above: FAIL (`stationary_noise_like`).
    double spectral_flatness_warn = 0.72;  ///< Above: BORDERLINE (`borderline_noisy_speech`).
    double min_band_mid = 0.10;            ///< Below: FAIL (no voice-band energy).
    double max_band_high = 0.70;           ///< Above: FAIL (hiss / high-band noise).
    FrameConfig frame;                     ///< Spectral analysis framing.
};

/// @brief Compute GateMetrics for @p chunk and apply the rules in order.
///
/// Rules are ordered cheapest-first and the first match decides:
/// energy floor, silence, clipping, frame activity, flatness (alone and with
/// ZCR), voice-band presence, high-band dominance, then the two BORDERLINE
/// rules; otherwise PASS. Metrics are always fully computed so downstream
/// consumers (scene, learned features) see every value.
GateResult evaluate_chunk(const Chunk& chunk, const GateConfig& cfg);

/// @}

}  // namespace edge_audio
