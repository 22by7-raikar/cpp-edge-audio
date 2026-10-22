/// @file
/// @brief Mono float PCM buffers, WAV loading and sample-rate conversion.
/// @ingroup audio

#pragma once

#include <string>
#include <vector>

namespace edge_audio {

/// @defgroup audio Audio input
/// @brief Getting audio into the runtime as mono float PCM.
///
/// Two entry points: load_wav() for complete files (offline pipeline) and
/// RawPcmDecoder for unbounded S16LE streams (streaming pipeline). Both produce
/// mono `float` samples in [-1, 1]; everything downstream assumes that format.
/// @{

/// @brief A contiguous block of mono float32 PCM in [-1, 1].
struct AudioBuffer {
    std::vector<float> samples;  ///< Mono samples, normalised to [-1, 1].
    int sample_rate = 0;         ///< Sample rate in Hz; 0 means empty.
    int channels = 1;            ///< Always 1: multi-channel input is downmixed on load.

    /// @brief Duration in seconds, or 0 for an empty buffer.
    double duration_sec() const {
        if (sample_rate == 0 || samples.empty()) return 0.0;
        return static_cast<double>(samples.size()) / sample_rate;
    }
};

/// @brief Load a WAV file as mono float PCM.
///
/// Any integer or float WAV encoding supported by dr_wav is accepted.
/// Multi-channel audio is downmixed by averaging channels; the sample rate is
/// preserved (see resample()).
///
/// @param path  Path to the WAV file.
/// @param out   Receives the decoded audio on success; unspecified on failure.
/// @param error Receives a human-readable message on failure.
/// @return `true` on success.
bool load_wav(const std::string& path, AudioBuffer& out, std::string& error);

/// @brief Resample with linear interpolation.
///
/// Adequate for speech front-ends feeding ASR; it applies no anti-aliasing
/// filter, so it is not suitable for high-fidelity conversion.
///
/// @param in          Source audio.
/// @param target_rate Output sample rate in Hz.
/// @return A new buffer at @p target_rate (a copy of @p in if the rates match).
AudioBuffer resample(const AudioBuffer& in, int target_rate);

/// @}

}  // namespace edge_audio
