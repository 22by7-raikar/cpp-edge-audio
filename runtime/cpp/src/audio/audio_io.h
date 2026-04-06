#pragma once

#include <string>
#include <vector>

namespace pipeline {

// Flat mono float32 PCM buffer, normalized to [-1.0, 1.0].

/*
---PCM: This is the standard method for digitally representing analog audio signals. 
        It captures the "amplitude" (volume) of the sound wave at regular intervals. 

---Pulse Code Modulation (PCM) buffer is a temporary storage area in memory used to hold raw, 
        uncompressed digital audio data before it is processed or played

---Mono: Mono audio has a single channel, meaning all sound is mixed together and played through one channel. 
        In contrast to stereo audio, which has two channels (left and right); more immersive sound experience.

---Flat: Flat audio means the samples are stored in a contiguous block of memory, without any additional structure or metadata.
        This allows for efficient processing and manipulation of the audio data.
*/

struct AudioBuffer {
    std::vector<float> samples;
    int sample_rate = 0;    // Hz
    int channels    = 1;    // always 1 after load (downmixed)

    double duration_sec() const {
        if (sample_rate == 0 || samples.empty()) return 0.0;
        return static_cast<double>(samples.size()) / sample_rate;
    }
};

// Load a WAV file into a mono float32 AudioBuffer.
// Multi-channel audio is downmixed to mono by averaging channels.
// Returns true on success; error message written to out_error on failure.
bool load_wav(const std::string& path, AudioBuffer& out, std::string& out_error);
// Linear-interpolation resample to target_rate.
// Adequate for speech; not broadcast-quality.
AudioBuffer resample(const AudioBuffer& in, int target_rate);
}  // namespace pipeline
