# AGENTS.md

Purpose:
Build and maintain a Linux-first, C++-leaning, deployment-oriented audio ML/STT pipeline.

Repository priorities:
1. C++ runtime engineering
2. inference optimization
3. FFT/audio-feature gating
4. reproducible benchmarking
5. deployment-minded structure

Current structure:
- vendor/whisper.cpp/: ASR backend
- tools/python/: support tooling only
- runtime/cpp/: current home for native runtime code
- ios-app/: not current priority
- docs/: only update when necessary

Global rules:
- Keep changes incremental and runnable
- Prefer simple working code over speculative architecture
- Use whisper.cpp unless explicitly told otherwise
- Do not generate iOS, Swift, Xcode, or macOS-specific code unless explicitly requested
- Do not move Python into the main runtime path
- Keep files focused and modular
- Preserve current run/build commands when refactoring
- Update docs only when setup or architecture actually changes

Strict output rules:
- Never use emojis
- Do not create extra markdown docs unless explicitly requested
- Do not create placeholder summaries, roadmap files, migration notes, or design docs unless explicitly asked
- Keep prose minimal

Current technical direction:
- Build Linux-first
- Support file-based and microphone-based audio ingestion
- Add chunking
- Add FFT/audio-feature gating
- Transcribe accepted chunks with whisper.cpp
- Log timings and metrics
- Benchmark deployment tradeoffs
