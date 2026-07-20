# Quality Model Status

## Executive Summary

Two different learned-quality experiments are present.

- `artifacts/quality_model/` is the only tracked artifact set. It is the repository-authoritative artifact, but it is the older 2026-05-21 run. It trained on the then-current 1,700-row `quality_train.jsonl`, whose speech source was LibriSpeech `test-clean`, and used the 450-row `eval_subset.jsonl` both for threshold selection and reported evaluation. Its selected threshold is 0.3.
- `artifacts/quality_model_clean/` is an untracked, later 2026-05-23 run at commit `6261978`. It is the source of the 4.0% FAR, 5.6% FRR, and 95.55% F1 claim. It trained after `quality_train.jsonl` was regenerated from LibriSpeech `train-clean-100`; its selected threshold is 0.4. It is a useful experimental candidate, but it is not authoritative because it is untracked, its validation/test arguments were not saved, and the current splits contain duplicate source-derived examples.
- Both joblib files contain `sklearn.ensemble.GradientBoostingClassifier` models with 100 estimators, depth 3, learning rate 0.1, random state 42, and 27 input features. No scaling, class weighting, or resampling is used for the saved GBT.
- One model input row represents one complete labeled audio file. The file is divided into non-overlapping 5-second chunks, the final short chunk is retained, 12 chunk metrics are aggregated by mean and maximum across all chunks, and three rule-gate decision fractions are appended. Correct integration therefore requires an end-of-file or end-of-utterance accumulator, not independent scoring of each streaming chunk.
- The 4.0% FAR, 5.6% FRR, and 95.55% F1 values are reproduced exactly on `eval_subset` at the classifier's implicit 0.5 decision threshold: TP=236, FP=8, TN=192, FN=14. They are not the metrics for the saved selected threshold 0.4. At threshold 0.4 on the same examples, the clean model reproduces TP=242, FP=13, TN=187, FN=8, FAR=6.5%, FRR=3.2%, and F1=95.84%.
- Leakage is present in the current split files. Current `quality_train` shares 87 deterministic music/noise examples with `quality_val`, 80 with `quality_test`, and 10 with `eval_subset`. `quality_val` and `eval_subset` also share 161 LibriSpeech utterance IDs and 56 deterministic derived examples. The older committed model's historical training labels have no source or base-utterance overlap with `eval_subset`, but its threshold was selected on that evaluation set.
- The learned model is Python-only. The C++ runtime neither loads the joblib artifact nor contains exported trees, constructs the 27-feature vector, produces a learned probability, or exposes learned/hybrid admission policies. Native ASR admission is currently rule gate plus rule-based scene/adaptive policy.

## Repository State

Initial state:

| Item | Value |
|---|---|
| Branch | `neura/quality-model-audit` |
| HEAD | `efc397e Merge pull request #1 from 22by7-raikar/sync/latest-local-neura` |
| `README.md` | Absent |
| Python environment declaration | `environment.yml`: Python 3.11, NumPy, SoundFile, scikit-learn; versions are not pinned |

Initial untracked paths reported by `git status --short`:

```text
?? .gitignore.save
?? artifacts/quality_model_clean/
?? data/manifests/demand_16k.jsonl
?? data/manifests/musan_music.jsonl
?? data/manifests/musan_noise.jsonl
?? data/manifests/musan_speech.jsonl
?? data/manifests/rirs.jsonl
```

All eight files under `artifacts/quality_model/` are tracked. All eight files under `artifacts/quality_model_clean/` are untracked. Files under `benchmarks/results/` are local ignored files due to `.gitignore`.

Final validation after creating this report:

- `git diff --stat` is empty because no tracked file was modified.
- The pre-existing untracked paths above remain present.
- The only new path created by this audit is `reports/quality_model_status.md`.

## Artifact Inventory

The committed artifact files have checkout mtimes of 2026-07-19 14:08:29 -0500, but their embedded run ID is `20260521_015515`. The clean files have both mtime and embedded run ID corresponding to 2026-05-23 02:06:01 -0500.

| Exact path | State | Bytes | Run/commit | Apparent purpose |
|---|---:|---:|---|---|
| `artifacts/quality_model/quality_gbt.joblib` | tracked | 142,403 | `20260521_015515` / `961ae9d` | Serialized GBT and `scaler=None`; SHA-256 `6c286f516fde8353cdbdd5bb0b12e8be72452febf7f854369ee8b0278fb72909` |
| `artifacts/quality_model/feature_schema.json` | tracked | 1,406 | same | Ordered feature schema, hyperparameters, threshold, and label-path metadata |
| `artifacts/quality_model/operating_point.json` | tracked | 211 | same | Max-F1 sweep row at threshold 0.3 |
| `artifacts/quality_model/metrics.json` | tracked | 1,033 | same | Default `predict()` metrics on `eval_subset`; mislabeled with selected threshold 0.3 |
| `artifacts/quality_model/threshold_sweep.tsv` | tracked | 456 | same | GBT sweep at thresholds 0.1 through 0.9 on `eval_subset` |
| `artifacts/quality_model/feature_importance.json` | tracked | 787 | same | 27 GBT impurity importances |
| `artifacts/quality_model/false_accepts.csv` | tracked | 1,863 | same | 18 `eval_subset` false accepts at threshold 0.3 with probabilities |
| `artifacts/quality_model/false_rejects.csv` | tracked | 1,414 | same | 11 `eval_subset` false rejects at threshold 0.3 with probabilities |
| `artifacts/quality_model_clean/quality_gbt.joblib` | untracked | 142,547 | `20260523_020601` / `6261978` | Serialized later GBT and `scaler=None`; SHA-256 `b59bad7f87226ba4fc0c191aeffef2fcb85db8eb1827dc5f722388a40eb129f8` |
| `artifacts/quality_model_clean/feature_schema.json` | untracked | 1,406 | same | Same feature order and hyperparameters, selected threshold 0.4 |
| `artifacts/quality_model_clean/operating_point.json` | untracked | 212 | same | Max-F1 sweep row at threshold 0.4 on a 1,700-example tuning split |
| `artifacts/quality_model_clean/metrics.json` | untracked | 1,030 | same | Default 0.5 `predict()` metrics on `eval_subset`; mislabeled with selected threshold 0.4 |
| `artifacts/quality_model_clean/threshold_sweep.tsv` | untracked | 482 | same | GBT sweep at thresholds 0.1 through 0.9 on the 1,700-example tuning split |
| `artifacts/quality_model_clean/feature_importance.json` | untracked | 782 | same | Later model's 27 impurity importances |
| `artifacts/quality_model_clean/false_accepts.csv` | untracked | 964 | same | 13 `eval_subset` false accepts at threshold 0.4 with probabilities |
| `artifacts/quality_model_clean/false_rejects.csv` | untracked | 771 | same | 8 `eval_subset` false rejects at threshold 0.4 with probabilities |

## Comparison of quality_model and quality_model_clean

| Property | `quality_model` | `quality_model_clean` |
|---|---|---|
| Version-control status | Tracked | Untracked |
| Run ID | `20260521_015515` | `20260523_020601` |
| Recorded git commit | `961ae9d` | `6261978` |
| Model hash | `6c286f...2909` | `b59bad...129f8` |
| Saved model class | `GradientBoostingClassifier` | Same |
| Ordered schema | Same 27 names | Same 27 names |
| Model parameters | 100 trees, depth 3, LR 0.1, seed 42 | Same |
| Training-label path string | `/home/apr/.../quality_train.jsonl` | Same string |
| Actual training-label content | Historical 1,700-row test-clean-derived file | Regenerated 1,700-row train-clean-100-derived file |
| Eval-label path string | `/home/apr/.../eval_subset.jsonl` | Same string |
| Selected threshold | 0.3 | 0.4 |
| Sweep/tuning dataset | `eval_subset`, 250 yes/200 no | 1,700 examples, 1,000 yes/700 no; consistent with `quality_val`, but the argument was not persisted |
| `operating_point.json` | FAR .09, FRR .044, F1 .9428 | FAR .17, FRR .063, F1 .9115 |
| `metrics.json` actual threshold | 0.5 | 0.5 |
| `metrics.json` counts | TP 230, FP 13, TN 187, FN 20 | TP 236, FP 8, TN 192, FN 14 |

The directories are unequivocally different experiments: joblib hashes and sizes differ, run IDs and git commits differ, selected thresholds differ, feature importances differ, and prediction/confusion results differ. The identical label-path strings are insufficient provenance because the content at `quality_train.jsonl` changed between runs. Neither schema records label-file hashes, validation labels, test labels, the full invocation, or package versions.

The only defensible use of "authoritative" is scoped:

- Repository/distribution authority: `artifacts/quality_model/`, because it is tracked.
- Source of the 95.55% claim: `artifacts/quality_model_clean/`, because its reloaded predictions reproduce that metric.
- Deployment or leakage-free evaluation authority: neither artifact.

## Training Implementation and Hyperparameters

The training entry point is `tools/python/training/train_quality_model.py`.

Candidate models evaluated on all 27 features are:

1. `sklearn.linear_model.LogisticRegression`: `C=1.0`, `solver="liblinear"`, `max_iter=500`, `random_state=42`; trained on standardized features.
2. `sklearn.ensemble.RandomForestClassifier`: `n_estimators=100`, `max_depth=6`, `min_samples_leaf=3`, `n_jobs=-1`, `random_state=42`; raw features.
3. `sklearn.ensemble.GradientBoostingClassifier`: `n_estimators=100`, `max_depth=3`, `learning_rate=0.1`, `random_state=42`; raw features.

The final artifact model is hard-coded to GBT; the code does not dynamically choose the best candidate. The reloaded full GBT parameters also show `loss="log_loss"`, `subsample=1.0`, `min_samples_leaf=1`, `min_samples_split=2`, and 100 fitted estimators. `GradientBoostingClassifier` has no configured class weighting here, and the code performs no over/under-sampling. The current training distribution is 1,000 positive and 700 negative examples.

The source-documented command is:

```bash
python tools/python/training/train_quality_model.py \
  --train-labels data/labels/quality_train.jsonl \
  --labels data/labels/eval_subset.jsonl \
  --out benchmarks/results/quality_model/
```

The current source additionally supports `--val-labels`, `--test-labels`, `--save-artifact`, and `--eval-artifact`. No saved file records the exact command used for the clean run, so an executed command including those options cannot be claimed verbatim.

## Exact Ordered Feature Schema

The exact ordered input is:

```text
 1. rms_mean
 2. silence_ratio_mean
 3. clipping_ratio_mean
 4. zcr_mean
 5. flatness_mean
 6. centroid_hz_mean
 7. rolloff_hz_mean
 8. flux_mean
 9. band_low_mean
10. band_mid_mean
11. band_high_mean
12. active_frac_mean
13. rms_max
14. silence_ratio_max
15. clipping_ratio_max
16. zcr_max
17. flatness_max
18. centroid_hz_max
19. rolloff_hz_max
20. flux_max
21. band_low_max
22. band_mid_max
23. band_high_max
24. active_frac_max
25. gate_pass_frac
26. gate_borderline_frac
27. gate_fail_frac
```

The 12 underlying chunk metrics are `rms`, `silence_ratio`, `clipping_ratio`, `zcr`, `flatness`, `centroid_hz`, `rolloff_hz`, `flux`, `band_low`, `band_mid`, `band_high`, and `active_frac`.

The Python-to-C++ field mappings are `flatness` to `spectral_flatness`, `centroid_hz` to `spectral_centroid`, `rolloff_hz` to `spectral_rolloff`, `flux` to `spectral_flux`, the three `band_*` values to the three `band_energy_*` values, and `active_frac` to `active_frame_frac`.

## Feature Aggregation and Inference Granularity

Construction is nested:

1. The complete audio file is loaded and mixed to mono.
2. It is divided into non-overlapping 5.0-second chunks. A final partial chunk is retained.
3. Time-domain chunk metrics are calculated over samples: RMS, silence ratio, clipping ratio, and crossings per second.
4. Each chunk is also split into 512-sample frames with a 256-sample hop. Hann-windowed FFT features are calculated per frame. Flatness, centroid, 85% rolloff, flux, and low/mid/high band fractions are averaged within the chunk; active fraction is the fraction of frames with RMS at least 0.005.
5. The rule gate assigns PASS, BORDERLINE, or FAIL to each chunk using the default gate configuration.
6. For each of the 12 chunk metrics, the mean and maximum across every chunk in the file are emitted: 24 values.
7. PASS, BORDERLINE, and FAIL counts are divided by the number of chunks: three more values.

This is not aggregation over VAD segments. The inner spectral computation is frame-based, but the learned row is file/utterance-based. It is not a rolling streaming window. One entire labeled file produces one 27-value row and one `should_transcribe` target.

The saved eval feature artifact confirms that `eval_subset` produced 450 rows: 390 files had one chunk, 46 had two, 8 had three, 4 had four, and 2 had six. Only clean-speech source files produced multiple chunks; the generated classes were one 5-second chunk each.

Correct native integration of these exact artifacts requires collecting rule-gate metrics for every non-overlapping 5-second chunk belonging to a defined utterance/file, computing the 24 aggregate statistics and three fractions when that utterance ends, then running one GBT inference. Per-chunk probability gating would change the feature distribution and inference granularity and would require a separately designed and evaluated model.

## Dataset and Split Summary

Current available label files:

| Label file | Intended speech source | Rows | Yes / No | Per-label counts | Unique paths | Unique source paths | Unique non-empty base IDs |
|---|---|---:|---:|---|---:|---:|---:|
| `data/labels/quality_train.jsonl` | LibriSpeech train-clean-100 | 1,700 | 1,000 / 700 | clean 400; noise speech 400; reverb 200; music 200; stationary 150; clipped 200; low utility 150 | 1,700 | 1,676 | 1,326 |
| `data/labels/quality_val.jsonl` | LibriSpeech dev-clean | 1,700 | 1,000 / 700 | same 1,700-class distribution | 1,700 | 1,460 | 1,110 |
| `data/labels/quality_test.jsonl` | LibriSpeech test-clean | 1,700 | 1,000 / 700 | same 1,700-class distribution | 1,700 | 1,448 | 1,098 |
| `data/labels/eval_subset.jsonl` | LibriSpeech dev-clean | 450 | 250 / 200 | clean 100; noise speech 100; reverb 50; music 50; stationary 50; clipped 50; low utility 50 | 450 | 429 | 329 |

Rows with non-empty `base_utterance_id` are 1,350 in each quality split and 350 in `eval_subset`. Repeated base IDs within a split are expected when multiple corruptions derive from the same speech utterance.

Artifact-specific protocol:

- Committed run: `quality_train.jsonl` at commit `961ae9d` had 1,700 rows with the same class counts but used LibriSpeech test-clean for speech-derived examples. It had 1,469 unique source paths and 1,119 unique base IDs. The 450-row untracked-at-that-commit `eval_subset` was used as `--labels`. Validation/test options did not yet exist in the training source.
- Clean run: the schema records `quality_train.jsonl` and `eval_subset.jsonl`. At its recorded commit, `quality_train` had been regenerated from train-clean-100 and `quality_val`/`quality_test` existed. The sweep denominators prove a 1,700-row, 1,000-positive/700-negative tuning input. Source chronology and the new `--val-labels` protocol make `quality_val.jsonl` the likely tuning file, but the filename was not saved. Whether `quality_test.jsonl` was passed to `--test-labels` is not recoverable from artifacts; held-out metrics were only printed, not persisted.

## Leakage Analysis

No exact derived output path is shared across current splits, and no split contains duplicate deterministic derived fingerprints internally. Source identity and content overlap nevertheless exist:

| Current split pair | Shared base IDs | Shared source identities | Deterministically duplicate derived examples |
|---|---:|---:|---:|
| train vs validation | 0 | 87 | 87: 64 music, 23 stationary noise |
| train vs test | 0 | 80 | 80: 62 music, 18 stationary noise |
| train vs eval_subset | 0 | 10 | 10: 6 music, 4 stationary noise |
| validation vs test | 0 | 113 | 113: 92 music, 21 stationary noise |
| validation vs eval_subset | 161 | 194 | 56: 20 clean, 3 low utility, 26 music, 7 stationary noise |
| test vs eval_subset | 0 | 48 | 48: 40 music, 8 stationary noise |

A deterministic fingerprint uses label plus normalized source and the transformation arguments that determine output: noise source and SNR for speech-in-noise, RIR for reverb, clip threshold for clipped audio, and source alone for clean, music, stationary-noise, and low-utility construction.

The current builder has a portability-related exclusion bug. `load_excluded_sources()` keeps label-source strings as stored, while `load_manifest()` resolves repo-relative manifest paths to absolute paths. The music/noise builders compare these strings directly. Relative excluded sources therefore fail to match absolute manifest sources, allowing duplicate music/noise examples across output splits.

Additional findings:

- The historical committed-run training set has zero source-identity and zero base-ID overlap with `eval_subset`; its builder exclusion worked with the then-absolute paths.
- The committed run still has threshold-selection leakage: threshold 0.3 was chosen by maximizing F1 on the same `eval_subset` used to report the operating point.
- The clean run's current train/validation and train/test duplicates compromise threshold selection and any held-out test claim.
- Validation and `eval_subset` both use dev-clean and share 161 utterance IDs. If threshold 0.4 is assessed on `eval_subset`, that assessment is not independent of validation.

## Path Portability Findings

Current tracked label files and tracked LibriSpeech manifests use repo-relative paths. The following machine-specific paths remain:

- Tracked historical metadata: absolute paths occur in `artifacts/quality_model/feature_schema.json`, `false_accepts.csv`, and `false_rejects.csv`. They do not prevent loading the joblib or using a caller-supplied eval label file, but they make error-analysis paths and training provenance machine-specific.
- Untracked clean metadata: `artifacts/quality_model_clean/feature_schema.json` stores absolute train/eval label paths. Its false-decision CSVs are repo-relative.
- Untracked manifests: all rows in `demand_16k.jsonl`, `musan_music.jsonl`, `musan_noise.jsonl`, `musan_speech.jsonl`, and `rirs.jsonl` contain `/home/apr/...` paths. Counts are 272, 660, 930, 426, and 60,417 occurrences respectively. These files will not work on another checkout without regeneration or normalization and are also involved in the exclusion mismatch above.
- Ignored historical results: the gate baseline/parity JSON files contain absolute evaluated-audio paths; quality-model comparison JSON files contain the absolute training-label path. The baseline's embedded metrics remain usable without audio, but rerunning path-based analysis elsewhere would fail.

No other explicit `/Users`, `/mnt`, `/opt`, `/srv`, `/tmp`, `/var`, `/data`, `/workspace`, or `/root` machine path was found in the audited artifact/label/result data.

## Threshold and Operating-Point Selection

Implemented learned thresholds are 0.1 through 0.9 in 0.1 increments. For each probability model:

- `balanced` is the sweep row with maximum F1.
- `conservative` is the maximum-F1 row subject to `FRR <= rule_gate_FRR + 0.02`, falling back to minimum FRR.
- If validation labels are supplied, the sweep uses validation. Otherwise it uses the primary `--labels` eval data.
- If test labels are supplied, final GBT test metrics use the validation-balanced threshold, but are printed only.
- Candidate model comparison and the `gbt_result` saved into `metrics.json` use `clf.predict()`, which is the implicit 0.5 threshold.
- `feature_schema.json`, `operating_point.json`, artifact reload, and false-decision CSV generation use the selected balanced threshold.

Saved thresholds:

| Run | Sweep data | Balanced | Conservative | Final-test record |
|---|---|---:|---:|---|
| committed | `eval_subset` itself | 0.3 | 0.3 | none; val/test support did not exist |
| clean | 1,700-row tuning set, strongly consistent with `quality_val` | 0.4 | 0.3 | none persisted; actual `--test-labels` use unknown |

The clean sweep's selected row is TP=937, FP=119, TN=581, FN=63 on a 1,000-positive/700-negative tuning set, giving FAR .17, FRR .063, precision .8873, recall .937, and F1 .9115.

Use of held-out data for threshold selection:

- Committed artifact: yes. `eval_subset` was both tuning and evaluation data.
- Clean artifact: no direct proof of test-based selection. The result is consistent with intended validation selection, but the validation filename is missing. Even under that intended protocol, duplicate examples contaminate train/validation and validation/eval.

`operating_point.json` and `metrics.json` disagree because the artifact writer saves an operating-point row from the probability sweep but saves `gbt_result['metrics']`, which was computed earlier with `clf.predict()` at 0.5. It then writes the selected threshold into the top-level `metrics.json` metadata without recomputing those metrics at that threshold. The false-decision CSVs do recompute at the selected threshold.

## Metric Sources

The code uses:

```text
FAR       = FP / (FP + TN)
FRR       = FN / (FN + TP)
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
F1        = 2 * precision * recall / (precision + recall)
```

Metric trace:

| Values | Source and interpretation |
|---|---|
| FAR .0400, FRR .0560, F1 .9555 | Clean GBT, `eval_subset`, implicit threshold 0.5; TP 236, FP 8, TN 192, FN 14. Saved in clean `metrics.json` and `model_comparison_20260523_020601.json`. |
| FAR .1700, FRR .0630, F1 .9115 | Clean GBT, selected threshold 0.4, 1,700-example tuning sweep. Saved in clean `operating_point.json` and sweep. |
| FAR .0650, FRR .0320, F1 .9584 | Clean GBT, selected threshold 0.4, `eval_subset`; recomputed from the existing feature artifact. The 13 FA and 8 FR rows are saved, but this aggregate is not saved. |
| FAR .0900, FRR .0440, F1 .9428 | Committed GBT, selected threshold 0.3, `eval_subset`; TP 239, FP 18, TN 182, FN 11. Saved operating point and reproduced. |
| FAR .0650, FRR .0800, F1 .9331 | Committed GBT, implicit threshold 0.5, `eval_subset`; TP 230, FP 13, TN 187, FN 20. Saved in `metrics.json`. |
| FAR .4250, FRR .0320, F1 .8388 | Default Python rule gate, same 450 `eval_subset` files and file-level accept/reject mapping. TP 242, FP 85, TN 115, FN 8. |

No saved final held-out `quality_test` confusion matrix was found.

## Metric Recalculation and Artifact Reload

System `python3` is Python 3.8.10 and lacks scikit-learn. The existing `audio_king` environment is Python 3.11.15 with NumPy 2.4.4, scikit-learn 1.8.0, joblib 1.5.3, and SoundFile 0.13.1. Both joblibs embed scikit-learn 1.8.0.

Both artifacts loaded successfully in that environment with no warnings. Each payload is a dictionary containing `clf` and `scaler`; `scaler` is `None`, `clf` has 27 input features and classes `[0, 1]`, and all 100 estimators are present.

Predictions were recomputed without reading raw audio. `benchmarks/results/gate_calibration/baseline_20260520_153342.json` contains the 450 `eval_subset` files and their rounded chunk metrics/decisions under the exact default gate configuration. Reapplying the training aggregation produced a `(450, 27)` float32 matrix with SHA-256 `98666a9c3ce7e5bec22fa8d8f46e993e25e015083a67686f8d1f831b9db740f5`.

Results:

| Artifact/threshold | TP | FP | TN | FN | FAR | FRR | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| committed / 0.5 | 230 | 13 | 187 | 20 | .0650 | .0800 | .9465 | .9200 | .9331 |
| committed / selected 0.3 | 239 | 18 | 182 | 11 | .0900 | .0440 | .9300 | .9560 | .9428 |
| clean / 0.5 | 236 | 8 | 192 | 14 | .0400 | .0560 | .9672 | .9440 | .9555 |
| clean / selected 0.4 | 242 | 13 | 187 | 8 | .0650 | .0320 | .9490 | .9680 | .9584 |

Agreement with saved per-error predictions is exact:

- Committed: all 18 false accepts and all 11 false rejects matched by normalized path; every probability matched after rounding to four decimals.
- Clean: all 13 false accepts and all 8 false rejects matched; every rounded probability matched.

Complete saved probability vectors do not exist, so agreement can only be measured for saved error rows plus the aggregate confusion metrics. Model training was not rerun. Cross-version joblib compatibility remains a concern because `environment.yml` does not pin scikit-learn 1.8.0; current exact-version reload succeeds, but a future environment may warn or fail.

## Rule Gate Versus Learned Model

An apples-to-apples comparison is available in each `model_comparison` result at the file level on the same 450 `eval_subset` examples:

| Model | Decision definition | FAR | FRR | F1 |
|---|---|---:|---:|---:|
| default rule gate | File accept if any chunk PASS or BORDERLINE | .4250 | .0320 | .8388 |
| clean GBT at default 0.5 | File probability at least 0.5 | .0400 | .0560 | .9555 |

This comparison has the same labels, examples, file-level aggregation, and metric formulas. It is still not an independent generalization result because ten negative eval examples are deterministic duplicates of current training examples. The learned feature vector also incorporates rule-gate decision fractions, so the learned model is built partly on the baseline's output.

The calibrated-rule artifact reports FAR .385 and FRR .032, but its FAR counts only `should_transcribe=no` files whose file decision is PASS. BORDERLINE files are excluded from FAR even though its accept rate includes PASS plus BORDERLINE and the runtime admits both. Under the learned-model binary definition, the calibrated configuration's 77 negative PASS and 7 negative BORDERLINE decisions would mean 84/200 admitted negatives, or .42, not .385. It also reports no precision/recall/F1. Therefore calibrated-rule .385/.032 and learned .04/.056/.9555 are not an apples-to-apples comparison.

The native runtime is another non-equivalent condition: it evaluates and admits individual fixed or VAD-derived chunks and, by default, changes rule thresholds using prior scene history. The Python learned model evaluates a complete file with static default gate features.

## Current C++ Runtime Behavior

The active native path is:

1. `main.cpp` loads a WAV and resamples it to 16 kHz.
2. Default mode uses fixed chunks (`chunk_ms=5000`, no overlap, final partial chunk retained). `--vad-asr` uses energy/ZCR VAD segments; `--vad-asr-packed` pads and merges them into ASR windows.
3. Before each chunk, the adaptive controller may tighten the rule-gate RMS/flatness thresholds based on preceding scene labels.
4. `evaluate_chunk()` calculates the 12 chunk metrics and applies ordered, rule-based PASS/BORDERLINE/FAIL thresholds.
5. The rule-based scene classifier labels the same chunk metrics as silence, noise, music, speech, mixed, or unknown.
6. ASR runs when gate is disabled or the chunk is PASS/BORDERLINE, unless adaptive policy suppresses a silence/music scene.
7. `AsrEngine::transcribe()` calls whisper.cpp `whisper_full()` on that chunk.

The C++ runtime:

- does not load either quality joblib;
- does not contain exported GBT trees;
- does not aggregate the 12 metrics into the ordered 27-feature vector across chunks;
- does not calculate a learned probability;
- has no rule/learned/hybrid CLI or policy setting;
- uses `--model` only for the whisper.cpp ASR model.

Python training is in `train_quality_model.py`. Python artifact evaluation is the `--eval-artifact` mode in that same file. Native DSP/rule inference is under `runtime/cpp/src/gate`, and actual ASR admission is the `run_asr` Boolean in `main.cpp`. These are separate paths; the Python model does not influence native admission.

## Verified Claims

- The saved learned model class is `GradientBoostingClassifier`.
- It accepts exactly 27 ordered features formed from 12 chunk metrics, 12 across-chunk means, 12 maxima, and 3 gate-decision fractions.
- Both models have 100 estimators, maximum depth 3, learning rate 0.1, and random state 42.
- The approximately 95.55% F1, 4% FAR, and 5.6% FRR values are real and reproducible for the clean model on 450 `eval_subset` files at threshold 0.5.
- The clean artifact's selected probability threshold is 0.4, and the committed artifact's selected threshold is 0.3.
- Rule-gate calibration metrics can differ substantially, but definitions must be aligned before comparison.
- Learned inference is Python-only; it is not used by the C++ runtime.

## Unverified or Conflicting Claims

- "95.55% F1 at the selected 0.4 operating point" is false. The value is from implicit threshold 0.5; selected-threshold eval F1 is 95.84%, while validation operating-point F1 is 91.15%.
- `metrics.json`'s top-level threshold correctly describes neither metrics file. The confusion metrics were computed at 0.5.
- The exact clean-run validation and held-out test arguments are unverified because they were not persisted. The validation association is strong but inferential; held-out test use and metrics are unknown.
- No leakage-free clean test result is available. Current train/validation/test/eval files share deterministic examples.
- The clean artifact cannot be called repository-authoritative because it is untracked.
- The committed threshold-0.3 score is not held out from threshold selection.
- Reproducing predictions under scikit-learn versions other than 1.8.0 is unverified.

## Claims Safe to Make in an Interview

- "I built and audited a Python scikit-learn GBT prototype that predicts whether a complete labeled audio file is worth transcribing from 27 aggregated DSP/rule-gate features."
- "The prototype uses 100 depth-3 boosting stages and aggregates non-overlapping 5-second chunk metrics into one file-level decision."
- "On a 450-example eval subset, the later artifact reproduces TP=236, FP=8, TN=192, FN=14 at threshold 0.5, corresponding to 4.0% FAR, 5.6% FRR, and 95.55% F1."
- "An audit found split contamination and a threshold/metrics serialization bug, so I treat those numbers as prototype diagnostics rather than a final held-out benchmark."
- "The production C++ path still uses an interpretable rule gate, scene policy, and whisper.cpp; learned gate inference has not been integrated."

## Claims Not Yet Safe to Make

- "The model achieves 95.55% F1 on a leakage-free held-out test set."
- "The selected threshold 0.4 has 4% FAR and 5.6% FRR."
- "Validation, test, and eval are disjoint."
- "The clean artifact is reproducible from a fully recorded command and immutable dataset hashes."
- "The learned model runs in or controls the C++ runtime."
- "The learned model has been benchmarked in streaming production conditions."

## Recommended Next Implementation Step

First repair dataset/provenance correctness before native integration: canonicalize source identities before split exclusion, add a pre-write cross-split duplicate assertion, regenerate disjoint train/validation/test labels, and save immutable label hashes plus validation/test filenames and the exact invocation in the artifact schema. In the same training-artifact workflow, recompute `metrics.json` at the recorded threshold and persist full per-example probabilities or a hashed 27-feature table. Only after a leakage-free held-out result is reproduced should this file-level GBT be exported to C++, or a separate per-window model be designed for streaming admission.
# Correction: leakage-controlled Neura V1 run (2026-07-20 CDT)

The remediation requested after this audit produced `artifacts/quality_model_neura_v1/` without modifying either prior artifact directory. Its train, validation, and held-out test label files each contain 1,700 examples (1,000 positive and 700 negative). Pairwise canonical-source, base-utterance, derived-fingerprint, output-path, and cheap content-hash overlap counts are all zero.

The fixed GBT was trained only on `quality_train`. Validation alone selected threshold 0.3. At that threshold, validation is TP=979, FP=153, TN=547, FN=21, FAR=.2186, FRR=.0210, precision=.8648, recall=.9790, and F1=.9184. The frozen-threshold held-out test is TP=986, FP=157, TN=543, FN=14, FAR=.2243, FRR=.0140, precision=.8626, recall=.9860, and F1=.9202. A fresh process reloaded the artifact and reproduced all 1,700 test decisions exactly with maximum and mean probability differences of 0.0.

This result corrects the split leakage, path normalization, threshold-metric labeling, and provenance gaps described above. The older 95.55% F1 / 4.0% FAR / 5.6% FRR result remains a valid reproduction of the contaminated legacy diagnostic at implicit threshold 0.5, but it is not comparable to this leakage-controlled held-out test. Full protocol and results are in `reports/quality_model_neura_v1.md`.
