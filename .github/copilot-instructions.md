# Repository-wide Copilot instructions

This repository is for a Linux-first, C++-focused, deployment-oriented audio ML project.

Current project priority:
- Build the core runtime on Ubuntu/Linux first
- Use whisper.cpp as the ASR backend
- Focus on C++ runtime engineering, inference optimization, deployment structure, and benchmarking
- Keep Python limited to support tooling, offline analysis, evaluation, plotting, and utilities
- Do not prioritize iOS, Swift, Xcode, or macOS-specific work unless explicitly requested

Core project goals:
- audio ingestion from files and microphone
- chunking and segmentation
- lightweight pre-ASR gating
- FFT/audio-feature-based quality filtering
- Whisper-based transcription
- structured logging and benchmarking
- optional later extensions such as scene/context classification and adaptive inference control

Code generation rules:
- Keep code modular, small, and runnable
- Do not invent fake APIs
- Do not create large speculative abstractions
- Generate only the files needed for the current milestone
- Prefer incremental implementation over full-project scaffolding
- Maintain bit allignment and memory safety in C++

Style rules:
- Never use emojis anywhere
- Do not create extra markdown docs unless explicitly requested
- Do not add decorative formatting
- Keep comments concise and practical
- Keep explanations short and technical

Architecture rules:
- Main runtime should become C++ as soon as practical
- Python is not the main runtime
- Separate audio I/O, chunking, feature extraction, gate logic, ASR wrapper, and logging
- Start with a non-learned, interpretable gate before any learned gate
- Design for benchmarkability and deployment
- Preserve a path to future optional side-models, but do not prioritize them now

Gate requirements:
The gate should be lightweight and interpretable.
Features to support now or soon:
- chunk duration
- RMS energy
- silence ratio
- clipping ratio
- zero-crossing rate
- spectral flatness
- spectral centroid
- spectral rolloff
- spectral flux
- normalized band-energy ratios

Gate outputs:
- PASS
- FAIL
- BORDERLINE
- reason string
- metrics struct

Benchmarking rules:
Treat benchmarking as a first-class feature.
Make it easy to compare:
- model variant
- quantization setting if applicable
- thread count
- chunk size
- gate enabled vs disabled
- latency
- real-time factor
- acceptance/rejection rate

Strict non-goals right now:
- no iOS app work
- no SwiftUI
- no Xcode integration
- no cloud-first implementation
- no diarization
- no speaker separation
- no emotion recognition as a main feature
- no fancy GUI
- no unnecessary docs
