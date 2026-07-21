# Quality Gate Runtime Integration

## Scope and artifact identity

The native `quality_model_neura_v1` classifier is integrated into the Linux C++ WAV runtime without changing the trained model, generated trees, ordered feature schema, DSP formulas, labels, or default rule thresholds.

- Model SHA-256: `045481f2ca42739495e814573c2575cb5cfd8520d521b4a4d5ba2df4d4f4358b`
- Metadata SHA-256: `32192c38fe6c53620635b0acd5719ed34054d09ff72eb06886da3ebc30f94c9f`
- Generated header SHA-256: `26b5d38e5166b21c08e1e3a2fb9b61026ce3e3723ec248bbe2ffff01598c0617`
- Schema: `quality-file-features-v1`
- Default learned threshold: `0.3`

## Architecture and utterance boundary

For `learned` and `hybrid`, one complete input WAV is one model example:

```text
complete WAV -> resample to 16 kHz -> non-overlapping 5 s analysis chunks
             -> existing 12 DSP metrics + base gate result per chunk
             -> 12 means + 12 maxima + PASS/BORDERLINE/FAIL fractions
             -> one ordered 27-float vector -> one native prediction
             -> final policy decision -> zero or one whole-WAV Whisper call
```

The original resampled WAV remains resident until the final decision. Accepted learned/hybrid input is passed to the existing `AsrEngine::transcribe()` once as the complete, contiguous resampled WAV. Analysis chunks are not transcribed and are not concatenated into another buffer.

Input-mode boundaries are explicit:

- Fixed chunks with `rule`: unchanged; each eligible chunk is sent independently to ASR.
- `--vad-asr` with `rule`: unchanged; each raw VAD segment is an independent gate/ASR chunk.
- `--vad-asr-packed` with `rule`: unchanged; each existing padded/merged VAD window is independent.
- `learned` and `hybrid`: only the complete-WAV, fixed 5-second, zero-overlap analysis boundary is supported. VAD, packed-VAD, custom chunk/hop, disabled gate, or non-training DSP/rule settings are rejected instead of inventing a segment-recombination policy.
- Live microphone streaming remains unsupported. A separately specified utterance segmentation/end-of-utterance policy is required before learned streaming can be enabled.

## Policy definitions

- `rule` is the default. The existing gate, scene classifier, adaptive thresholds, music/silence suppression, and per-chunk ASR scheduling are decision-equivalent to the pre-integration path.
- `learned` aggregates the completed WAV, runs the native classifier once, and admits when `probability >= threshold`. The rule summary is logged but does not affect admission.
- `hybrid` is a conservative heuristic, not a trained hybrid model. The file is admitted only when the current rule scheduler would admit at least one analysis chunk and the learned classifier admits the file. Both decisions are logged.

The learned vector always uses base training-time gate decisions. Adaptive rule decisions are calculated separately for the hybrid rule summary and never change the learned feature vector.

## CLI

```bash
# Existing behavior; --quality-policy is optional.
runtime/cpp/build_tests/audio_pipeline \
  --input input.wav --model model.bin

runtime/cpp/build_tests/audio_pipeline \
  --input input.wav --model model.bin \
  --quality-policy learned --quality-threshold 0.3

runtime/cpp/build_tests/audio_pipeline \
  --input input.wav --model model.bin \
  --quality-policy hybrid --quality-threshold 0.3

# Emit the ordered values only when explicitly requested.
runtime/cpp/build_tests/audio_pipeline \
  --input input.wav --gate-only --quality-policy learned \
  --quality-debug-features
```

Invalid policy names and non-finite/out-of-range thresholds fail before audio processing. The active policy, threshold, and complete-WAV boundary are printed at startup.

## Aggregation and parity

`QualityFeatureAggregator` accepts the existing 12 `GateMetrics` values and one gate result per chunk. It reproduces the authoritative Python record pipeline by rounding each chunk metric to six decimal places, then emitting the 12 means, 12 maxima, and three gate fractions in the frozen order. Empty aggregation throws, reset clears all state, and non-finite inputs fail without partially updating the accumulator.

Native unit fixtures verify every index, means, maxima, three fractions, single/multiple chunks, the final short chunk, Python six-decimal record behavior, empty/reset behavior, and non-finite rejection.

Parity evidence:

- Exported model corpus: 3,400/3,400 decisions agree; maximum raw-score difference `0`; maximum probability difference `1.1102230246251565e-16`.
- Native end-to-end feature extraction, aggregation, and prediction: 1,300/1,300 available held-out WAV decisions agree with the frozen saved predictions at threshold `0.3`.
- On the 11-second JFK smoke input, upstream C++ versus Python DSP values differed by at most `2.44140625e-4`; this did not alter the decision. Aggregation arithmetic and ordering are exact, but upstream cross-language DSP is not claimed bit-identical.

## Logging

Each input emits an additive `event=quality_summary` TSV record and an additive JSON `quality` object containing:

- policy and analysis chunk count
- feature schema version
- learned raw score, probability, inference time, threshold, and decision
- current rule summary and final admission
- whether ASR ran, ASR time/result, and rejection reason

Rule mode records learned values as `not_run`/JSON `null`. The 27 values are excluded by default and included only with `--quality-debug-features`.

## Evaluation

### Full frozen held-out feature table

This comparison covers all 1,700 held-out examples using the immutable saved feature table, saved learned probabilities, and saved PASS/BORDERLINE rule baseline. ASR invocations are one file-level call per admitted example; ASR was not executed. The saved rule baseline does not include the C++ scene/adaptive suppression layer.

| Policy | TP | FP | TN | FN | FAR | FRR | F1 | ASR invocations | Calls avoided |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Rule baseline | 957 | 416 | 284 | 43 | 0.5943 | 0.0430 | 0.8066 | 1,373 | 327 (19.24%) |
| Learned | 986 | 157 | 543 | 14 | 0.2243 | 0.0140 | 0.9202 | 1,143 | 557 (32.76%) |
| Hybrid heuristic | 944 | 127 | 573 | 56 | 0.1814 | 0.0560 | 0.9116 | 1,071 | 629 (37.00%) |

Decision-table load, verification, and comparison took `21.88 ms` locally.

### Native held-out WAV subset

The native evaluator processed all 1,300 WAV examples (6,500 seconds) and skipped 400 clean-speech FLAC inputs that the WAV-only runtime cannot load. It exercised native audio loading, DSP, gate/scene/adaptive logic, aggregation, and native model inference. ASR invocations are scheduled-call counts; Whisper itself was not run.

| Policy | TP | FP | TN | FN | FAR | FRR | F1 | ASR invocations | Calls avoided |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Rule runtime | 334 | 263 | 437 | 266 | 0.3757 | 0.4433 | 0.5581 | 597 | 703 (54.08%) |
| Learned | 588 | 157 | 543 | 12 | 0.2243 | 0.0200 | 0.8743 | 745 | 555 (42.69%) |
| Hybrid heuristic | 328 | 81 | 619 | 272 | 0.1157 | 0.4533 | 0.6501 | 409 | 891 (68.54%) |

The native evaluation took `5.722 s`, a gate real-time factor of `0.000880`. The current scene/adaptive rule path rejects many positive examples on this subset; the hybrid truth table deliberately inherits those rejects. Hybrid should therefore be treated as a conservative compute-saving option, not as a quality improvement over the learned policy.

## Performance and memory

Release measurements on this machine:

- Aggregation: `43.7 ns/chunk` over 600,000 chunk additions; accumulator size `224 bytes`.
- Native GBT: mean `893.9 ns`, p50 `862 ns`, p95 `902 ns`, about 1.12 million predictions/second.
- Compiled model data: `36,112 bytes`.
- Native WAV-subset gate evaluation: `5.722 s` for 6,500 seconds of audio.
- Accepted whole-file Whisper tiny.en smoke: `416.8-438.9 ms` for the 11-second WAV, one call. Rule mode used three calls totaling `1,174.2 ms` on the same file. ASR compute dominates aggregation and learned inference by more than five orders of magnitude.

The current file runtime already retains the complete `AudioBuffer` and copies it into fixed chunks. Learned/hybrid add no PCM buffer beyond that existing layout. At 16 kHz mono float32:

| File length | Original buffer | Existing chunk copies | Total existing PCM | Added learned PCM |
|---|---:|---:|---:|---:|
| 5 s | 0.305 MiB | 0.305 MiB | 0.610 MiB | 0 |
| 30 s | 1.831 MiB | 1.831 MiB | 3.662 MiB | 0 |
| 60 s | 3.662 MiB | 3.662 MiB | 7.324 MiB | 0 |

The non-PCM learned state is the 224-byte accumulator plus 36,112 bytes of static model data.

## Validation

- `bash scripts/run_tests.sh Release`: four CTest targets passed, zero failures.
- Focused runtime suite: 14 passed, zero failed.
- Native model parity: 3,400/3,400 decisions, zero disagreements.
- Rule default regression: all three JFK chunk log records were identical before and after integration.
- Real ASR smoke: `rule`, `learned`, and `hybrid` completed; learned/hybrid each ran ASR once when admitted at threshold `0`.
- Invalid policy and invalid threshold CLI smoke tests returned nonzero with explicit errors.
- `git diff --check`: clean.

## Limitations

- Learned/hybrid policies are file-only and WAV-only in the current runtime.
- VAD and packed-VAD remain rule-only until a trained and validated utterance reconstruction boundary is specified.
- Live microphone input has no reliable learned-model utterance boundary and remains unsupported.
- The full 1,700-row comparison is table-based; the native audio-path comparison excludes 400 FLAC clean-speech files.
- Held-out ASR was not run. Reported evaluation invocations are calls that policy scheduling would make.
- The heuristic hybrid is not learned or calibrated and has a high false-reject rate under the current scene/adaptive rule behavior.
