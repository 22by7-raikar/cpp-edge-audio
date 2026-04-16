// audio_io.cpp
// WAV file loading via dr_wav (public domain single-header, vendored at src/audio/dr_wav.h).
// Produces mono float32 PCM normalized to [-1.0, 1.0].

#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

#include "audio_io.h"

#include <algorithm>
#include <cstring>

namespace pipeline {

bool load_wav(const std::string& path, AudioBuffer& out, std::string& out_error) {
    drwav wav;
    if (!drwav_init_file(&wav, path.c_str(), nullptr)) {
        out_error = "dr_wav: failed to open: " + path;
        return false;
    }

    const uint32_t channels    = wav.channels;
    const uint32_t sample_rate = wav.sampleRate;
    const uint64_t frame_count = wav.totalPCMFrameCount;

    if (frame_count == 0) {
        drwav_uninit(&wav);
        out_error = "WAV file has zero frames: " + path;
        return false;
    }

    // Read interleaved float32 samples (all channels)
    std::vector<float> raw(frame_count * channels);
    const uint64_t read = drwav_read_pcm_frames_f32(&wav, frame_count, raw.data());
    drwav_uninit(&wav);

    if (read != frame_count) {
        out_error = "dr_wav: expected " + std::to_string(frame_count) +
                    " frames, got " + std::to_string(read);
        return false;
    }

    out.sample_rate = static_cast<int>(sample_rate);
    out.channels    = 1;
    out.samples.resize(frame_count);

    if (channels == 1) {
        out.samples = std::move(raw);
    } else {
        // Downmix: average across channels per frame
        const float inv_ch = 1.0f / static_cast<float>(channels);
        for (uint64_t i = 0; i < frame_count; ++i) {
            float sum = 0.0f;
            for (uint32_t c = 0; c < channels; ++c) {
                sum += raw[i * channels + c];
            }
            out.samples[i] = sum * inv_ch;
        }
    }

    return true;
}

AudioBuffer resample(const AudioBuffer& in, int target_rate) {
    if (in.sample_rate == target_rate || in.samples.empty()) return in;

    AudioBuffer out;
    out.sample_rate = target_rate;
    out.channels    = 1;

    const double ratio      = static_cast<double>(in.sample_rate) / target_rate;
    const size_t out_frames = static_cast<size_t>(in.samples.size() / ratio);
    out.samples.resize(out_frames);

    for (size_t i = 0; i < out_frames; ++i) {
        const double pos   = i * ratio;
        const size_t idx0  = static_cast<size_t>(pos);
        const size_t idx1  = std::min(idx0 + 1, in.samples.size() - 1);
        const float  frac  = static_cast<float>(pos - idx0);
        out.samples[i]     = in.samples[idx0] * (1.0f - frac) + in.samples[idx1] * frac;
    }

    return out;
}

}  // namespace pipeline
