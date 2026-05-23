// vad.cpp
// Frame-level DSP VAD: energy + ZCR classification with hangover and segment merging.

#include "vad.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace pipeline {

namespace {

// RMS of a frame (inline; loop vectorises cleanly)
inline float frame_rms(const float* s, int n) {
    float acc = 0.0f;
    for (int i = 0; i < n; ++i) acc += s[i] * s[i];
    return std::sqrt(acc / static_cast<float>(n));
}

// Zero-crossing rate in crossings/second
inline float frame_zcr(const float* s, int n, int sample_rate) {
    int crossings = 0;
    for (int i = 1; i < n; ++i) {
        if ((s[i] >= 0.0f) != (s[i-1] >= 0.0f)) ++crossings;
    }
    return static_cast<float>(crossings) * static_cast<float>(sample_rate)
           / static_cast<float>(n);
}

}  // namespace

std::vector<VadSegment> run_vad(
    const float*     samples,
    int              n_samples,
    int              sample_rate,
    const VadConfig& cfg)
{
    if (n_samples <= 0 || sample_rate <= 0) return {};

    const int frame_samp = cfg.frame_ms * sample_rate / 1000;
    const int hop_samp   = cfg.hop_ms   * sample_rate / 1000;

    if (frame_samp <= 0 || hop_samp <= 0) return {};

    // Number of complete frames (trailing partial frames are discarded).
    const int n_frames = (n_samples >= frame_samp)
                         ? (n_samples - frame_samp) / hop_samp + 1
                         : 0;
    if (n_frames <= 0) return {};

    // -----------------------------------------------------------------
    // Step 1 — raw per-frame classification
    //   speech = RMS >= energy_thresh  AND  ZCR < zcr_max
    // -----------------------------------------------------------------
    std::vector<bool> raw_speech(static_cast<size_t>(n_frames), false);
    for (int f = 0; f < n_frames; ++f) {
        const int offset = f * hop_samp;
        // offset + frame_samp <= n_samples is guaranteed by n_frames computation
        const float rms = frame_rms(samples + offset, frame_samp);
        const float zcr = frame_zcr(samples + offset, frame_samp, sample_rate);
        raw_speech[static_cast<size_t>(f)] = (rms >= cfg.energy_thresh) && (zcr < cfg.zcr_max);
    }

    // -----------------------------------------------------------------
    // Step 2 — hangover
    //   Once a raw-speech frame fires, hold speech state for
    //   hangover_frames additional frames (prevents fragmented segments).
    // -----------------------------------------------------------------
    std::vector<bool> smooth(static_cast<size_t>(n_frames), false);
    {
        int hold = 0;
        for (int f = 0; f < n_frames; ++f) {
            if (raw_speech[static_cast<size_t>(f)]) hold = cfg.hangover_frames + 1;
            if (hold > 0) {
                smooth[static_cast<size_t>(f)] = true;
                --hold;
            }
        }
    }

    // -----------------------------------------------------------------
    // Step 3 — extract contiguous speech segments from smooth labels
    // -----------------------------------------------------------------
    struct Seg { int s; int e; };  // half-open [s, e) in frame indices
    std::vector<Seg> segs;
    {
        int f = 0;
        while (f < n_frames) {
            if (!smooth[static_cast<size_t>(f)]) { ++f; continue; }
            const int s = f;
            while (f < n_frames && smooth[static_cast<size_t>(f)]) ++f;
            segs.push_back({s, f});
        }
    }
    if (segs.empty()) return {};

    // -----------------------------------------------------------------
    // Step 4 — merge adjacent segments whose silence gap is short
    // -----------------------------------------------------------------
    const int min_sil_samp   = cfg.min_silence_ms * sample_rate / 1000;
    const int min_sil_frames = std::max(1, min_sil_samp / hop_samp);

    std::vector<Seg> merged;
    merged.push_back(segs[0]);
    for (size_t i = 1; i < segs.size(); ++i) {
        if (segs[i].s - merged.back().e <= min_sil_frames) {
            merged.back().e = segs[i].e;
        } else {
            merged.push_back(segs[i]);
        }
    }

    // -----------------------------------------------------------------
    // Step 5 — filter short segments and compute output
    // -----------------------------------------------------------------
    const int min_sp_samp   = cfg.min_speech_ms * sample_rate / 1000;
    const int min_sp_frames = std::max(1, min_sp_samp / hop_samp);

    std::vector<VadSegment> result;
    for (const auto& seg : merged) {
        const int len = seg.e - seg.s;
        if (len < min_sp_frames) continue;

        // Count raw-speech frames to compute speech_ratio (excludes hangover-only frames)
        int raw_count = 0;
        for (int ff = seg.s; ff < seg.e; ++ff) {
            if (raw_speech[static_cast<size_t>(ff)]) ++raw_count;
        }

        VadSegment vs;
        vs.start_sec  = static_cast<double>(seg.s * hop_samp) / sample_rate;
        // End boundary: end of the last frame in the segment, capped at signal length
        vs.end_sec    = std::min(
            static_cast<double>((seg.e - 1) * hop_samp + frame_samp) / sample_rate,
            static_cast<double>(n_samples) / sample_rate);
        vs.frame_count  = len;
        vs.speech_ratio = (len > 0) ? static_cast<float>(raw_count) / static_cast<float>(len)
                                    : 0.0f;
        result.push_back(vs);
    }

    return result;
}

}  // namespace pipeline
