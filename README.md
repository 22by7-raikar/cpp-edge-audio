# edge_audio

[![ci](https://github.com/22by7-raikar/cpp-edge-audio/actions/workflows/ci.yml/badge.svg)](https://github.com/22by7-raikar/cpp-edge-audio/actions/workflows/ci.yml)

A C++17 runtime for on-device speech recognition that **decides whether audio
is worth transcribing before spending ASR compute on it**, plus a streaming
voice-command loop built on the same decision. ASR is
[whisper.cpp](https://github.com/ggerganov/whisper.cpp) on CPU or CUDA.

On an edge device a Whisper call costs hundreds of milliseconds, while the
admission decision here costs microseconds. Music, noise, silence and clipped
audio are rejected first, using DSP features and a gradient-boosted model
trained in Python and compiled into the binary.

## Results

| | |
|---|---|
| Admission quality, held-out test (1,700 files) | learned F1 **0.920** vs rule gate 0.807; recall 98.6% |
| ASR calls avoided vs transcribing everything | **32.8%** learned · 37.0% hybrid |
| Native model vs scikit-learn | **3,400 / 3,400** identical decisions, max probability error 1.1e-16 |
| Learned model cost | ~1 µs per decision, 36 KB of `constexpr` data, no allocation |
| Streaming buffers | bounded by the 8 s utterance cap (~0.5 MB), independent of stream length |

Evaluation protocol, splits, leakage checks and caveats:
[docs/quality_model.md](docs/quality_model.md).

## Architecture

![System overview](docs/diagrams/system_overview.svg)

- **`audio_pipeline`**: offline. Admits a complete WAV under a `rule`,
  `learned` or `hybrid` policy, optionally transcribes it, and emits
  per-window records plus a JSON report.
- **`voice_command_stream`**: streaming. Reads raw PCM on stdin, endpoints
  utterances, admits each one, transcribes it at most once, and writes one
  JSON command event per utterance.

Both are thin wrappers over `edge_audio_core`, a dependency-free library
(audio I/O, segmentation, quality admission, command routing, reporting). Only
`edge_audio_asr` links whisper.cpp. Design, contracts and trade-offs are in
[docs/architecture.md](docs/architecture.md); the API reference is generated
with Doxygen.

## Quick start

```bash
git clone --recursive https://github.com/22by7-raikar/cpp-edge-audio.git
cd cpp-edge-audio
cmake --preset release && cmake --build --preset release -j
ctest --preset release                  # 100 tests, no model weights needed
scripts/download_model.sh base.en       # -> models/ggml-base.en.bin
```

**Offline admission and transcription**

```bash
build/release/apps/audio_pipeline --input vendor/whisper.cpp/samples/jfk.wav \
    --model models/ggml-base.en.bin --policy learned --report jfk.json
```

Omit `--model` to analyse without transcribing. `scripts/compare_policies.sh`
runs one file through all three policies.

**Streaming voice commands**

```bash
scripts/voice_commands.sh --model models/ggml-base.en.bin --mic          # live (ALSA)
scripts/voice_commands.sh --model models/ggml-base.en.bin --wav cmds.wav  # replay
```

One event per utterance (abridged; full schema in
[architecture.md](docs/architecture.md#voice-commands-command)):

```json
{"event":"voice_command","utterance_id":1,"duration_ms":1120,"finalization_reason":"end_silence",
 "quality_policy":"rule","admitted":true,"rejection_reason":"","asr_ran":true,
 "transcript":"Look left!","intent":"LOOK_LEFT","action":"look_left",
 "latency":{"quality_us":...,"asr_ms":...,"post_utterance_ms":...},"stream":{"frames_processed":85}}
```

Requirements: CMake ≥ 3.21, a C++17 compiler (CI: GCC 13 on Ubuntu 24.04), Python 3
for the protocol test, ffmpeg for the helper scripts, and the CUDA toolkit for
`--preset cuda`.

## Command-line reference

| `audio_pipeline` | |
|---|---|
| `--input <wav>` | Required. Any rate and channel count; resampled to 16 kHz mono. |
| `--model <path>` | whisper.cpp model. Omit to analyse only. |
| `--policy rule\|learned\|hybrid` | Admission policy (default `rule`). |
| `--threshold <p>` | Learned threshold in [0, 1] (default 0.3, chosen on validation). |
| `--segmentation fixed\|vad\|vad-packed` | Windowing (default `fixed`, 5 s). Learned policies require `fixed`. |
| `--no-gate`, `--no-adapt` | Disable the rule gate / scene-adaptive control. |
| `--report <json>` | Also write a JSON report. |

| `voice_command_stream` | |
|---|---|
| `--rate <hz> --channels <1\|2>` | Required. Format of S16LE PCM on stdin. |
| `--model <path>` | Required. whisper.cpp model. |
| `--policy`, `--threshold` | As above (default `rule`). |
| `--capture-device`, `--transport` | Labels copied into every event. |

Exit status: 0 on success, 2 for invalid arguments (before any model loads),
1 for runtime failures.

## Repository layout

```
src/edge_audio/   audio · segmentation · quality · asr · command · report
apps/             audio_pipeline, voice_command_stream, evaluate_policies, benchmark_quality
tests/            GoogleTest suite, stream protocol test, model parity corpus
tools/python/     DSP feature mirror, training, export, policy evaluation
scripts/          dataset preparation, model download, demos
artifacts/        the trained quality model (quality_model_v1) and its evaluation
docs/             architecture, quality model, diagrams, Doxyfile
```

## Development

| Task | Command |
|---|---|
| Debug build + tests | `cmake --preset debug && cmake --build --preset debug && ctest --preset debug` |
| AddressSanitizer + UBSan | `cmake --preset asan && cmake --build --preset asan && ctest --preset asan` |
| CUDA build | `cmake --preset cuda && cmake --build --preset cuda` |
| Python tests | `python -m pytest tests/python` |
| API docs | `doxygen docs/Doxyfile` → `docs/api/html/index.html` |
| Diagrams | `docs/diagrams/render.sh` (Mermaid sources → SVG) |
| Format | `clang-format -i $(git ls-files 'src/*.h' 'src/*.cpp' 'apps/*.cpp' 'tests/*.cpp')` |
| Model latency | `build/release/apps/benchmark_quality` |

Presets build with `-Wall -Wextra -Wpedantic -Wshadow -Werror`. CI runs the
Release and sanitizer suites, the Python tests, a check that the parity corpus
is reproducible from the committed artifact, and clang-format.

Changing the model: retrain, then run
`python tools/python/export/export_quality_model.py`. This regenerates
`quality_model_data.h` and the parity corpus together, so the parity test
fails if the C++ and Python sides ever diverge.

## Limitations

- The learned model is trained on class-balanced, largely synthetic data. Its
  error rates are not field estimates, and music and stationary noise cause
  all of its false accepts.
- Probabilities are not calibrated for 1–2 s commands. Use `rule` for
  streaming until the model is recalibrated on command-length audio.
- The frame VAD is energy/ZCR based, and the streaming loop is
  single-threaded: ASR runs synchronously between pushes.
- Command routing is exact-match over nine phrases, by design. A
  mis-transcription yields `UNKNOWN`, never a different command.

## License

MIT. See [LICENSE](LICENSE). whisper.cpp (MIT) is a git submodule, and dr_wav
(public domain / MIT-0) is vendored in `third_party/`.
