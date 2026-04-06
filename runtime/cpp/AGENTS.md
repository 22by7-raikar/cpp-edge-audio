# C++ runtime agent rules

Scope:
This directory is for the main native runtime.

Goals:
- C++ runtime around whisper.cpp
- deployment-oriented design
- benchmarkable execution
- lightweight FFT/audio-feature gate
- modular, testable structure

Priorities:
1. audio loading / capture
2. chunking
3. framing + FFT feature extraction
4. gate decision logic
5. ASR wrapper
6. logging / benchmarking

Rules:
- Prefer standard C++ and CMake-friendly structure
- Keep dependencies minimal
- Avoid unnecessary abstraction layers
- Make configuration explicit: model path, threads, chunk size, thresholds
- Optimize for clear latency measurement and reproducible execution
- Never use emojis
- Do not create extra markdown docs unless explicitly requested
