# Quality-gate demo validation

Validation was run on branch `neura/demo-and-documentation` without downloading
models or datasets. The existing fixture was
`vendor/whisper.cpp/samples/jfk.wav` (11 seconds), and the existing local model
was `vendor/whisper.cpp/models/ggml-tiny.en.bin`.

## Fresh CPU Release build

The following commands used a newly created ignored directory:

```bash
fresh_build_dir=$(mktemp -d /tmp/cpp-edge-audio-demo-build.XXXXXX)
cmake -S runtime/cpp -B "$fresh_build_dir" \
  -DCMAKE_BUILD_TYPE=Release \
  -DWHISPER_ROOT="$PWD/vendor/whisper.cpp" \
  -DBUILD_TESTS=ON
cmake --build "$fresh_build_dir" -j2
(cd "$fresh_build_dir" && ctest --output-on-failure)
```

The created directory was `/tmp/cpp-edge-audio-demo-build.fP3NaM`. It configured
a CPU-only Release build and CTest passed all four tests: `pipeline_unit_tests`,
`quality_model_unit_tests`, `quality_gate_runtime_unit_tests`, and
`quality_model_cpp_parity`.

Focused checks were also run from that build:

```bash
"$fresh_build_dir/test_quality_gate_runtime"
"$fresh_build_dir/quality_model_parity" \
  runtime/cpp/tests/data/quality_model_neura_v1_parity.bin
```

The runtime test reported 14 passed, 0 failed. The parity tool reported
3,400/3,400 decisions agreed, maximum raw-score difference 0, maximum
probability difference `1.1102230246251565e-16`, model data size 36,112 bytes,
and 1,137,209 predictions/second in this run.

The repository-supported test and Python workflow commands also passed:

```bash
bash scripts/run_tests.sh Release
source /home/apr/miniconda3/etc/profile.d/conda.sh
conda activate audio_king
python -m pytest tests/python/test_quality_workflow.py -v
```

The native script's CTest run passed 4/4. The Python suite passed 8/8,
including `TrainingArtifactTest.test_metadata_and_artifact_hash_generation`.

## Direct policy demonstrations

Using the fresh binary and existing local assets:

```bash
WAV=vendor/whisper.cpp/samples/jfk.wav
MODEL=vendor/whisper.cpp/models/ggml-tiny.en.bin
BIN=/tmp/cpp-edge-audio-demo-build.fP3NaM/audio_pipeline

"$BIN" --input "$WAV" --model "$MODEL" --quality-policy rule
"$BIN" --input "$WAV" --model "$MODEL" \
  --quality-policy learned --quality-threshold 0.3 \
  --bench-json /tmp/quality-learned.json
"$BIN" --input "$WAV" --model "$MODEL" \
  --quality-policy hybrid --quality-threshold 0.3
"$BIN" --input "$WAV" --model "$MODEL" \
  --quality-policy learned --quality-threshold 0
```

`/tmp/quality-learned.json` parsed successfully with `python3 -m json.tool`.
For all four runs, the DSP rule admitted all three chunks and the learned score
was `-1.7736113904814736` (probability `0.14509379193850144`). Observed final
behavior matched the policy semantics:

| Invocation | Expected | Observed |
|---|---|---|
| `rule` | Rule admission runs Whisper once. | Admitted; `asr_ran=1`, 1,211.021 ms. |
| `learned`, threshold `0.3` | Learned score rejects; no Whisper. | Rejected as `learned_below_threshold`; `asr_ran=0`. |
| `hybrid`, threshold `0.3` | Rule admits but learned rejection vetoes; no Whisper. | Rejected as `hybrid_learned_rejected`; `asr_ran=0`. |
| `learned`, threshold `0` | Alternate CLI threshold admits and runs Whisper once. | Admitted; `asr_ran=1`, 481.881 ms. |

The alternate threshold is demonstration-only; it does not alter the recorded
validation-selected threshold of `0.3`.

## Demo wrapper

The wrapper was run with an explicit untracked output directory:

```bash
scripts/demo_quality_gate.sh \
  vendor/whisper.cpp/samples/jfk.wav \
  vendor/whisper.cpp/models/ggml-tiny.en.bin \
  /tmp/quality-gate-script-validation.b3DOU9/output
```

It made a CPU Release build below the output directory and produced per-policy
`*.tsv`, `*.json`, `*.stdout`, and `*.stderr` files. All three JSON files parsed
successfully. Rule admitted the fixture and ran Whisper once (1,155.252 ms).
Learned and hybrid each rejected at `0.3` without ASR, with rejection reasons
`learned_below_threshold` and `hybrid_learned_rejected` respectively.

## Skipped checks

- No model or dataset download was attempted.
- Real ASR was not skipped: the local tiny English model was available and was
  used for the rule and alternate-threshold learned demonstrations.
- CUDA was not validated. The documented optional CUDA configuration requires a
  newer CMake than this host's 3.16.3 when its CUDA backend is enabled; the
  required CPU build and demo were validated instead.
- No full dataset evaluation or training was rerun, because this task documents
  the frozen model and must not change recorded benchmark results.

## Final workspace check

`git diff --check` passed. Final `git status --short` was:

```text
 ? vendor/whisper.cpp
?? README.md
?? artifacts/quality_model_clean/
?? data/manifests/demand_16k.jsonl
?? data/manifests/musan_music.jsonl
?? data/manifests/musan_noise.jsonl
?? data/manifests/musan_speech.jsonl
?? data/manifests/rirs.jsonl
?? reports/demo_validation.md
?? scripts/demo_quality_gate.sh
```

The `vendor/whisper.cpp` state and the `artifacts/quality_model_clean/` and
`data/manifests/` entries existed before this documentation task. Within the
submodule, the existing untracked entry is `examples/python/__pycache__/`; no
vendor files were changed or staged by this work. No files under the frozen
model artifact directory, labels, generated native header, or runtime source
were changed.
