# Quality Model C++ Export and Parity

## Artifact identity

The native export uses only `artifacts/quality_model_neura_v1/`.

| Property | Value |
|---|---|
| Model class | `sklearn.ensemble.GradientBoostingClassifier` |
| scikit-learn | 1.8.0 |
| Model SHA-256 | `045481f2ca42739495e814573c2575cb5cfd8520d521b4a4d5ba2df4d4f4358b` |
| Metadata SHA-256 | `32192c38fe6c53620635b0acd5719ed34054d09ff72eb06886da3ebc30f94c9f` |
| Schema | `quality-file-features-v1` |
| Selected threshold | 0.3 |
| Features | 27 ordered file-level aggregate features |
| Estimators | 100 binary regression trees |
| Maximum depth | 3 |
| Exported nodes | 1,488 |
| Learning rate | 0.1 |
| Initial raw score | `0.35667494393873245` |

The exporter verifies the recorded hashes of the joblib, schema, operating point, validation matrix, and test matrix before writing output. It also requires the recorded sklearn version and exact model class, class order `[0, 1]`, feature count, estimator shape, hyperparameters, loss type, and prior initializer.

## sklearn formula

The saved classifier uses a `DummyClassifier(strategy="prior")` initializer. Its positive-class prior is `10/17`, so the binary initial raw score is the sklearn 1.8.0 logit-link result:

```text
initial_raw = log((10/17) / (7/17))
            = 0.35667494393873245
```

For input vector `x`, sklearn's dense gradient-boosting traversal follows the left child when `x[feature] <= threshold` and the right child otherwise. The raw positive-class score is:

```text
raw(x) = initial_raw + 0.1 * sum(tree_i_leaf_value(x), i=1..100)
```

The probability for `should_transcribe=yes` is:

```text
probability(x) = 1 / (1 + exp(-raw(x)))
```

The decision is `probability >= caller_threshold`. The native default is the artifact-selected threshold 0.3.

The implementation assumes the sklearn 1.8.0 binary `HalfBinomialLoss`, `LogitLink`, float32 tree inputs, float64 thresholds/leaf values, and estimator-order accumulation used by this artifact. The exporter stops if these assumptions do not match.

## Generated representation

`tools/python/export/export_quality_model.py` writes:

- `runtime/cpp/generated/quality_model_neura_v1.h`
- `runtime/cpp/tests/data/quality_model_neura_v1_parity.bin`

The generated header contains constexpr POD arrays with every tree's local left/right child indexes, feature indexes, thresholds, values, node ranges, learning rate, initial score, selected threshold, schema, ordered feature names, source run timestamp, sklearn version, and model/metadata hashes. It is generated from the joblib and must not be manually edited.

Two independent exporter runs were byte-identical:

| Output | Bytes | SHA-256 |
|---|---:|---|
| Generated header | 104,979 | `26b5d38e5166b21c08e1e3a2fb9b61026ce3e3723ec248bbe2ffff01598c0617` |
| Binary parity corpus | 506,720 | `0183560985d36d2ff045bb51d500b64d490ed84cc09f3b8721b462f337658114` |
| Compiled constexpr model data | 36,112 | N/A |

## Feature schema

One prediction consumes one completed file/utterance aggregation in this exact order:

```text
rms_mean
silence_ratio_mean
clipping_ratio_mean
zcr_mean
flatness_mean
centroid_hz_mean
rolloff_hz_mean
flux_mean
band_low_mean
band_mid_mean
band_high_mean
active_frac_mean
rms_max
silence_ratio_max
clipping_ratio_max
zcr_max
flatness_max
centroid_hz_max
rolloff_hz_max
flux_max
band_low_max
band_mid_max
band_high_max
active_frac_max
gate_pass_frac
gate_borderline_frac
gate_fail_frac
```

The native API takes `std::array<float, 27>` and calculates with double precision. A pointer/count overload rejects any count other than 27. NaN and positive or negative infinity are rejected with `std::invalid_argument` before traversal. This is a deliberate fail-fast policy; non-finite values are not silently routed through sklearn's tree behavior.

## Parity corpus and results

The deterministic fixture contains all 1,700 validation rows followed by all 1,700 held-out test rows from the saved NPZ feature tables. Each row contains the stable example ID, exact 27 float32 features, sklearn raw score, sklearn positive probability, and expected threshold-0.3 decision. Raw audio is not read.

| Metric | Result |
|---|---:|
| Examples | 3,400 |
| Maximum absolute raw-score difference | 0 |
| Mean absolute raw-score difference | 0 |
| Maximum absolute probability difference | `1.1102230246251565e-16` |
| Mean absolute probability difference | `2.6914362282976925e-18` |
| Binary decision agreement | 3,400 / 3,400 |
| Disagreements | 0 |
| Worst raw-score example ID | `482b0152fa85ffc6e5539c41` |
| Worst probability example ID | `d12f178cd3cb7f88a25818d3` |

The maximum probability difference is below the required `1e-10` tolerance without relaxing it.

## Native performance

Release build, 10,000 warm-up predictions followed by 100,000 timed predictions:

| Metric | Result |
|---|---:|
| Mean latency | 827.56 ns (0.828 us) |
| p50 latency | 822 ns |
| p95 latency | 861 ns |
| Predictions per second | 1,208,379 |

This is a correctness-oriented microbenchmark on one saved feature vector. It includes per-call clock measurement overhead and is machine-specific.

## Tests and commands

```bash
python tools/python/export/export_quality_model.py

bash scripts/run_tests.sh Release

runtime/cpp/build_tests/quality_model_parity \
  runtime/cpp/tests/data/quality_model_neura_v1_parity.bin
```

Native unit coverage includes left and right traversal, equality routing to the left child, direct leaf evaluation, ensemble accumulation, logistic conversion, the default and an alternate threshold, invalid feature count, null input, NaN/infinity rejection, full feature order, schema exposure, and model-hash exposure.

## Remaining limitations

- The API scores only an already completed 27-feature file/utterance aggregation. It does not build that aggregation from chunks.
- The learned model is not connected to `main.cpp`, VAD, ASR, scene policy, adaptive policy, or runtime admission.
- Export still requires the pinned Python/sklearn/joblib environment; production C++ inference does not invoke Python.
- The generated representation is specific to this binary sklearn GBT artifact and schema.
- The microbenchmark does not measure feature extraction, file aggregation, or end-to-end ASR latency.
