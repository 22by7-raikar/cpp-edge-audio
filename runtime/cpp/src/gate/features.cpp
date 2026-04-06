// features.cpp
// Frame-level FFT-based spectral feature extraction.
//
// FFT: in-place Cooley-Tukey radix-2 DIT (no external library required).
// All features are computed per-frame over a Hann-windowed segment.
// Chunk-level results are aggregates (mean, median, max) across frames.
//
// Input must be mono float32 PCM; sample_rate is used only for Hz conversion.
// Caller (gate.cpp) is responsible for ensuring the signal is at 16000 Hz.

#include "features.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <numeric>
#include <vector>

namespace pipeline {

// -------------------------------------------------------
// Utility: next-power-of-2 check
// -------------------------------------------------------
static bool is_power_of_2(int n) { return n > 0 && (n & (n - 1)) == 0; }

// -------------------------------------------------------
// Hann window coefficients (cached per frame_size)
// -------------------------------------------------------
static std::vector<float> make_hann(int n) {
    std::vector<float> w(n);
    for (int i = 0; i < n; ++i) {
        w[i] = 0.5f * (1.0f - std::cos(2.0f * static_cast<float>(M_PI) * i / (n - 1)));
    }
    return w;
}

// -------------------------------------------------------
// In-place Cooley-Tukey radix-2 DIT FFT.
// x.size() must be a power of 2.
// -------------------------------------------------------
static void fft_inplace(std::vector<std::complex<float>>& x) {
    const int n = static_cast<int>(x.size());

    // Bit-reversal permutation
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(x[i], x[j]);
    }

    // Butterfly stages
    for (int len = 2; len <= n; len <<= 1) {
        const float ang = -2.0f * static_cast<float>(M_PI) / len;
        const std::complex<float> w_step(std::cos(ang), std::sin(ang));
        for (int i = 0; i < n; i += len) {
            std::complex<float> w(1.0f, 0.0f);
            for (int j = 0; j < len / 2; ++j) {
                const auto u = x[i + j];
                const auto v = x[i + j + len / 2] * w;
                x[i + j]           = u + v;
                x[i + j + len / 2] = u - v;
                w *= w_step;
            }
        }
    }
}

// -------------------------------------------------------
// Compute magnitude spectrum (one-sided) from real input frame.
// Returns n/2 + 1 bins.
// -------------------------------------------------------
static std::vector<float> magnitude_spectrum(
    const float* frame, int n, const std::vector<float>& hann)
{
    std::vector<std::complex<float>> buf(n);
    for (int i = 0; i < n; ++i) {
        buf[i] = std::complex<float>(frame[i] * hann[i], 0.0f);
    }
    fft_inplace(buf);

    const int n_bins = n / 2 + 1;
    std::vector<float> mag(n_bins);
    for (int k = 0; k < n_bins; ++k) {
        mag[k] = std::abs(buf[k]);
    }
    return mag;
}

// -------------------------------------------------------
// Per-frame feature extraction
// -------------------------------------------------------
static FrameFeatures compute_frame_features(
    const std::vector<float>& mag,   // magnitude spectrum (n/2+1 bins)
    const std::vector<float>& prev_mag,
    const float* time_samples,
    int          n,
    int          sample_rate,
    const FrameConfig& cfg)
{
    FrameFeatures f;
    const int n_bins = static_cast<int>(mag.size());
    const float bin_hz = static_cast<float>(sample_rate) / static_cast<float>(cfg.frame_size);
    constexpr float kEps = 1e-10f;

    // ----- Frame RMS -----
    float rms_sq = 0.0f;
    for (int i = 0; i < n; ++i) rms_sq += time_samples[i] * time_samples[i];
    f.frame_rms = std::sqrt(rms_sq / n);

    // ----- Power spectrum -----
    std::vector<float> power(n_bins);
    float total_power = 0.0f;
    for (int k = 0; k < n_bins; ++k) {
        power[k]    = mag[k] * mag[k];
        total_power += power[k];
    }

    if (total_power < kEps) {
        // Silent frame – all spectral features remain 0
        return f;
    }

    // ----- Spectral flatness -----
    // = exp(mean(log(power))) / mean(power)
    double log_sum = 0.0;
    for (int k = 0; k < n_bins; ++k) {
        log_sum += std::log(static_cast<double>(power[k]) + kEps);
    }
    const double geom_mean  = std::exp(log_sum / n_bins);
    const double arith_mean = static_cast<double>(total_power) / n_bins;
    f.flatness = static_cast<float>(geom_mean / (arith_mean + kEps));
    f.flatness = std::min(f.flatness, 1.0f);  // clamp numerical overshoot

    // ----- Spectral centroid -----
    double weighted_sum = 0.0;
    for (int k = 0; k < n_bins; ++k) {
        weighted_sum += static_cast<double>(k) * power[k];
    }
    f.centroid_hz = static_cast<float>(weighted_sum / (total_power + kEps)) * bin_hz;

    // ----- Spectral rolloff -----
    const float rolloff_threshold = cfg.rolloff_percentile * total_power;
    float cumulative = 0.0f;
    f.rolloff_hz = static_cast<float>(n_bins - 1) * bin_hz;  // default: top bin
    for (int k = 0; k < n_bins; ++k) {
        cumulative += power[k];
        if (cumulative >= rolloff_threshold) {
            f.rolloff_hz = static_cast<float>(k) * bin_hz;
            break;
        }
    }

    // ----- Spectral flux (vs previous frame) -----
    if (!prev_mag.empty()) {
        float flux = 0.0f;
        for (int k = 0; k < n_bins; ++k) {
            const float diff = mag[k] - prev_mag[k];
            flux += diff * diff;
        }
        f.flux = flux;
    }

    // ----- Band energy ratios -----
    const int low_bin  = static_cast<int>(cfg.band_low_max_hz / bin_hz);
    const int mid_bin  = static_cast<int>(cfg.band_mid_max_hz / bin_hz);
    const int clamped_low = std::min(low_bin, n_bins - 1);
    const int clamped_mid = std::min(mid_bin, n_bins - 1);

    float e_low = 0.0f, e_mid = 0.0f, e_high = 0.0f;
    for (int k = 0;             k <= clamped_low; ++k) e_low  += power[k];
    for (int k = clamped_low+1; k <= clamped_mid; ++k) e_mid  += power[k];
    for (int k = clamped_mid+1; k < n_bins;       ++k) e_high += power[k];

    f.band_low  = e_low  / (total_power + kEps);
    f.band_mid  = e_mid  / (total_power + kEps);
    f.band_high = e_high / (total_power + kEps);

    return f;
}

// -------------------------------------------------------
// Median of a float vector (modifies copy)
// -------------------------------------------------------
static double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    const size_t mid = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + mid, v.end());
    if (v.size() % 2 == 1) return v[mid];
    const double lo = *std::max_element(v.begin(), v.begin() + mid);
    return (lo + v[mid]) * 0.5;
}

// -------------------------------------------------------
// Public entry point
// -------------------------------------------------------
ChunkFeatures extract_chunk_features(
    const float* samples,
    int          n_samples,
    int          sample_rate,
    const FrameConfig& cfg)
{
    ChunkFeatures result;

    if (!samples || n_samples <= 0 || sample_rate <= 0) return result;
    if (!is_power_of_2(cfg.frame_size)) return result;  // bad config

    const int frame_size = cfg.frame_size;
    const int hop_size   = cfg.hop_size > 0 ? cfg.hop_size : frame_size / 2;

    const std::vector<float> hann = make_hann(frame_size);

    std::vector<FrameFeatures> frames;
    std::vector<float> prev_mag;

    for (int offset = 0; offset + frame_size <= n_samples; offset += hop_size) {
        const float* frame_ptr = samples + offset;

        // Pad shorter final frame with zeros if it exists
        std::vector<float> frame_buf(frame_ptr, frame_ptr + frame_size);

        const auto mag = magnitude_spectrum(frame_buf.data(), frame_size, hann);

        FrameFeatures ff = compute_frame_features(
            mag, prev_mag, frame_buf.data(), frame_size, sample_rate, cfg);

        frames.push_back(ff);
        prev_mag = mag;
    }

    if (frames.empty()) return result;

    // ----- Aggregate -----
    result.n_frames = static_cast<int>(frames.size());

    std::vector<double> flatness_v, centroid_v;
    flatness_v.reserve(frames.size());
    centroid_v.reserve(frames.size());

    double flat_sum = 0, cent_sum = 0, roll_sum = 0;
    double flux_sum = 0, flux_max = 0;
    double bl_sum = 0, bm_sum = 0, bh_sum = 0;
    int    active_frames = 0;
    float  flat_max = 0.0f;

    for (const auto& f : frames) {
        flatness_v.push_back(f.flatness);
        centroid_v.push_back(f.centroid_hz);

        flat_sum += f.flatness;
        cent_sum += f.centroid_hz;
        roll_sum += f.rolloff_hz;
        flux_sum += f.flux;
        bl_sum   += f.band_low;
        bm_sum   += f.band_mid;
        bh_sum   += f.band_high;

        if (f.flux > flux_max)      flux_max  = f.flux;
        if (f.flatness > flat_max)  flat_max  = f.flatness;
        if (f.frame_rms >= cfg.active_rms_thresh) ++active_frames;
    }

    const double n = static_cast<double>(frames.size());
    result.flatness_mean    = flat_sum   / n;
    result.flatness_max     = flat_max;
    result.centroid_mean    = cent_sum   / n;
    result.centroid_median  = median(centroid_v);
    result.rolloff_mean     = roll_sum   / n;
    result.flux_mean        = flux_sum   / n;
    result.flux_max         = flux_max;
    result.band_low_mean    = bl_sum     / n;
    result.band_mid_mean    = bm_sum     / n;
    result.band_high_mean   = bh_sum     / n;
    result.active_frame_frac = static_cast<double>(active_frames) / n;

    return result;
}

}  // namespace pipeline
