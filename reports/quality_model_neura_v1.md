# Quality Model Neura V1

Run timestamp: 2026-07-20T06:14:49Z  
Source commit: `efc397e21160c9687668b22c65352cc2edca8851` (dirty worktree, as recorded before artifact creation)

## Protocol

`quality_model_neura_v1` is the corrected authoritative GBT experiment. The model was fitted only on `quality_train`, the probability threshold was selected only on `quality_val`, and the frozen threshold was applied once to `quality_test`. `eval_subset` was not used for model selection or final reporting; its canonical raw inputs were excluded during split construction so it remains a separate legacy diagnostic set.

Speech sources are LibriSpeech `train-clean-100`, `dev-clean`, and `test-clean` for train, validation, and test respectively. MUSAN music, MUSAN/DEMAND noise, and simulated RIR sources were deterministically partitioned before examples were rendered. The dataset seed was 123. The GBT uses 100 estimators, maximum depth 3, learning rate 0.1, and random state 42.

One model example is one complete audio file. Feature extraction divides the file into non-overlapping 5-second chunks, then forms the existing ordered 27-feature file vector: the mean and maximum of 12 DSP metrics plus three gate-decision fractions. No per-chunk or streaming model was introduced, and the C++ runtime is unchanged.

## Splits

| Split | Rows | Transcribe yes | Transcribe no | Clean | Speech + noise | Speech + reverb | Music | Stationary noise | Clipped | Low utility |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 1,700 | 1,000 | 700 | 400 | 400 | 200 | 200 | 150 | 200 | 150 |
| Validation | 1,700 | 1,000 | 700 | 400 | 400 | 200 | 200 | 150 | 200 | 150 |
| Test | 1,700 | 1,000 | 700 | 400 | 400 | 200 | 200 | 150 | 200 | 150 |

Label SHA-256 hashes:

- Train: `44dd52a8a0ed039c9dee7efb0b46a435ea0516a79c63dec0419725f4638a1b20`
- Validation: `044b662aaaab549faff6e968cead264bf089127f593e543d607b590260dea107`
- Test: `4494ea9ffcbe3183cf3bc2ebc88238ab0e8a4be9f431e607b37d50c1fa7d143b`

## Overlap validation

All mandatory pairwise checks passed after rendering. Content hashes cover cheaply hashable rendered regular files; clean-speech symlinks are represented by canonical source identity instead of following and rehashing raw audio.

| Pair | Canonical source | Base utterance | Derived fingerprint | Output path | Content hash |
|---|---:|---:|---:|---:|---:|
| Train / validation | 0 | 0 | 0 | 0 | 0 |
| Train / test | 0 | 0 | 0 | 0 | 0 |
| Validation / test | 0 | 0 | 0 | 0 | 0 |

## Selected operating point

The validation sweep used fixed thresholds 0.1 through 0.9 in increments of 0.1. Maximum validation F1 selected threshold 0.3. The test set did not participate in this choice.

| Split | Threshold | TP | FP | TN | FN | FAR | FRR | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Validation | 0.3 | 979 | 153 | 547 | 21 | 0.2186 | 0.0210 | 0.8648 | 0.9790 | 0.9184 |
| Held-out test | 0.3 | 986 | 157 | 543 | 14 | 0.2243 | 0.0140 | 0.8626 | 0.9860 | 0.9202 |

The separate 0.5 diagnostics are stored under explicitly named `default_threshold_0_5_diagnostic` objects. They are not labeled as selected-threshold metrics.

## Per-label results at threshold 0.3

| Split | Label | TP | FP | TN | FN |
|---|---|---:|---:|---:|---:|
| Validation | clean_speech | 391 | 0 | 0 | 9 |
| Validation | speech_in_noise | 393 | 0 | 0 | 7 |
| Validation | speech_in_reverb | 195 | 0 | 0 | 5 |
| Validation | music | 0 | 74 | 126 | 0 |
| Validation | stationary_noise | 0 | 61 | 89 | 0 |
| Validation | clipped_or_distorted | 0 | 18 | 182 | 0 |
| Validation | low_utility | 0 | 0 | 150 | 0 |
| Test | clean_speech | 398 | 0 | 0 | 2 |
| Test | speech_in_noise | 394 | 0 | 0 | 6 |
| Test | speech_in_reverb | 194 | 0 | 0 | 6 |
| Test | music | 0 | 82 | 118 | 0 |
| Test | stationary_noise | 0 | 56 | 94 | 0 |
| Test | clipped_or_distorted | 0 | 19 | 181 | 0 |
| Test | low_utility | 0 | 0 | 150 | 0 |

The held-out test has 157 false accepts and 14 false rejects. False accepts are 82 music, 56 stationary-noise, and 19 clipped examples. False rejects are 2 clean-speech, 6 speech-in-noise, and 6 speech-in-reverb examples. Full rows and probabilities are in `false_accepts_test.csv`, `false_rejects_test.csv`, and `test_predictions.csv`.

## Top feature importances

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | `flux_max` | 0.205080 |
| 2 | `rms_mean` | 0.130387 |
| 3 | `band_high_max` | 0.121762 |
| 4 | `clipping_ratio_mean` | 0.121250 |
| 5 | `flux_mean` | 0.094610 |
| 6 | `active_frac_max` | 0.065352 |
| 7 | `band_low_max` | 0.063443 |
| 8 | `clipping_ratio_max` | 0.039407 |
| 9 | `flatness_mean` | 0.025953 |
| 10 | `band_high_mean` | 0.020279 |

These are GBT impurity importances and should not be interpreted as causal effects.

## Artifact and reload verification

The artifact is under `artifacts/quality_model_neura_v1/`. Its model SHA-256 is `045481f2ca42739495e814573c2575cb5cfd8520d521b4a4d5ba2df4d4f4358b`. Validation and test feature matrices are saved as compressed NPZ tables and hashed in `model_metadata.json`. Both prediction CSVs contain 1,700 example rows with stable IDs, portable paths, targets, probabilities, the frozen threshold, decisions, and correctness.

A new Python process reloaded `quality_gbt.joblib`, recomputed probabilities from `test_features.npz`, and compared them with `test_predictions.csv`: all 1,700 binary decisions agreed exactly, maximum probability difference was 0.0, and mean probability difference was 0.0.

## Reproduction

```bash
python scripts/datasets/build_manifests.py --datasets musan rirs demand --validate
python scripts/datasets/build_quality_train.py --all-splits --overwrite --seed 123
python tools/python/training/train_quality_model.py \
  --authoritative-protocol \
  --train-labels data/labels/quality_train.jsonl \
  --val-labels data/labels/quality_val.jsonl \
  --test-labels data/labels/quality_test.jsonl \
  --save-artifact artifacts/quality_model_neura_v1 \
  --seed 42
```

The training command automatically performs the fresh-process reload verification. Exact argv, commit/dirty state, package versions, model parameters, label hashes, split summaries, code paths, and hashes for every non-metadata artifact file are recorded in `model_metadata.json`.

## Comparison with prior prototypes

The previous `quality_model_clean` claim of FAR 0.0400, FRR 0.0560, and F1 0.9555 was measured on the 450-row legacy `eval_subset` at an implicit threshold of 0.5, while its saved selected threshold was 0.4. Its split construction also contained source-derived overlap and did not preserve sufficient command, label-hash, validation, or test provenance. The new held-out test result of FAR 0.2243, FRR 0.0140, and F1 0.9202 is therefore not directly comparable: the dataset, split isolation, selection protocol, and decision threshold differ. The new result should replace the prior prototype as the leakage-controlled reference, not be presented as a regression against an equivalent benchmark.

## Remaining limitations

- The examples are class-balanced and substantially synthetic; they do not estimate production class prevalence or field error rates.
- Threshold selection uses a coarse fixed grid rather than a predeclared continuous calibration method.
- The model depends on the current Python DSP feature implementation and sklearn/joblib serialization environment.
- Music and stationary noise remain the dominant false-accept categories.
- No external microphone, device, language, or deployment-domain test set was evaluated.
- The learned model is not integrated into the C++ runtime.
