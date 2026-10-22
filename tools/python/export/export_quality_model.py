#!/usr/bin/env python3
"""Export the trained quality model to C++ and build the native parity corpus.

Two outputs, generated independently:

header  src/edge_audio/quality/quality_model_data.h
        The scikit-learn GradientBoostingClassifier as constexpr C++ tables.
        Requires the exact scikit-learn version recorded in the artifact metadata,
        because the joblib pickle is version-specific.

corpus  tests/data/quality_model_v1_parity.bin
        Every validation and test feature vector with the probability that
        scikit-learn produced at training time (recorded in *_predictions.csv).
        Needs only NumPy. The C++ test suite scores each vector natively and
        requires identical decisions and probabilities within 1e-12.

Usage:
    python tools/python/export/export_quality_model.py              # both
    python tools/python/export/export_quality_model.py --corpus-only
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import struct
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ARTIFACT_DIR = REPO_ROOT / "artifacts/quality_model_v1"
DEFAULT_HEADER = REPO_ROOT / "src/edge_audio/quality/quality_model_data.h"
DEFAULT_CORPUS = REPO_ROOT / "tests/data/quality_model_v1_parity.bin"
CORPUS_MAGIC = b"QMPAR2\0\0"
CORPUS_VERSION = 2
PARITY_SPLITS = ("validation", "test")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cpp_double(value: float) -> str:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"cannot export non-finite double: {value}")
    text = format(value, ".17g")
    if "." not in text and "e" not in text:
        text += ".0"
    return text


def _portable(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _load_authoritative(artifact_dir: Path):
    import joblib
    import sklearn
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import GradientBoostingClassifier

    required = [
        "quality_gbt.joblib",
        "feature_schema.json",
        "operating_point.json",
        "model_metadata.json",
        "validation_features.npz",
        "test_features.npz",
    ]
    missing = [name for name in required if not (artifact_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"authoritative artifact is incomplete: {missing}")

    metadata = json.loads((artifact_dir / "model_metadata.json").read_text())
    schema = json.loads((artifact_dir / "feature_schema.json").read_text())
    operating_point = json.loads(
        (artifact_dir / "operating_point.json").read_text()
    )
    model_path = artifact_dir / "quality_gbt.joblib"
    consumed_files = [
        "quality_gbt.joblib",
        "feature_schema.json",
        "operating_point.json",
        "validation_features.npz",
        "test_features.npz",
    ]
    for filename in consumed_files:
        computed_hash = _sha256(artifact_dir / filename)
        recorded_hash = metadata["artifact_file_hashes"][filename]
        if computed_hash != recorded_hash:
            raise RuntimeError(
                f"artifact hash mismatch for {filename}: "
                f"computed={computed_hash} recorded={recorded_hash}"
            )
    model_hash = _sha256(model_path)
    recorded_sklearn = metadata["package_versions"]["scikit-learn"]
    if sklearn.__version__ != recorded_sklearn:
        raise RuntimeError(
            "sklearn version mismatch: "
            f"running={sklearn.__version__} recorded={recorded_sklearn}"
        )

    payload = joblib.load(model_path)
    if sorted(payload) != ["clf", "scaler"] or payload["scaler"] is not None:
        raise RuntimeError("unexpected joblib payload or non-null scaler")
    clf = payload["clf"]
    if type(clf) is not GradientBoostingClassifier:
        raise RuntimeError(f"unexpected model class: {type(clf)!r}")
    if clf.classes_.tolist() != [0, 1]:
        raise RuntimeError(f"unexpected class order: {clf.classes_.tolist()}")
    if clf.n_features_in_ != 27:
        raise RuntimeError(f"unexpected feature count: {clf.n_features_in_}")
    if clf.estimators_.shape != (100, 1):
        raise RuntimeError(f"unexpected estimator shape: {clf.estimators_.shape}")
    expected_params = {
        "n_estimators": 100,
        "max_depth": 3,
        "learning_rate": 0.1,
        "random_state": 42,
    }
    actual_params = clf.get_params()
    for name, expected in expected_params.items():
        if actual_params[name] != expected:
            raise RuntimeError(
                f"unexpected model parameter {name}: {actual_params[name]}"
            )
    if max(tree.tree_.max_depth for tree in clf.estimators_[:, 0]) != 3:
        raise RuntimeError("unexpected maximum tree depth")
    if type(clf.init_) is not DummyClassifier or clf.init_.strategy != "prior":
        raise RuntimeError(f"unexpected initial estimator: {clf.init_!r}")
    if type(clf._loss).__name__ != "HalfBinomialLoss":
        raise RuntimeError(f"unexpected loss implementation: {type(clf._loss)!r}")

    feature_names = list(schema["feature_names"])
    if feature_names != metadata["ordered_feature_names"]:
        raise RuntimeError("feature schema and metadata order differ")
    if schema["schema_version"] != "quality-file-features-v1":
        raise RuntimeError(f"unexpected schema: {schema['schema_version']}")
    selected_threshold = float(operating_point["threshold"])
    if selected_threshold != float(schema["threshold"]) or selected_threshold != 0.3:
        raise RuntimeError("authoritative selected threshold is not 0.3")

    probe = np.zeros((2, clf.n_features_in_), dtype=np.float32)
    probe[1] = 1.0
    initial = clf._raw_predict_init(probe)[:, 0]
    if initial[0] != initial[1] or not np.isfinite(initial[0]):
        raise RuntimeError("initial raw prediction is not a finite constant")
    positive_prior = float(clf.init_.predict_proba(probe[:1])[0, 1])
    expected_initial = float(
        clf._loss.link.link(np.asarray([positive_prior], dtype=np.float64))[0]
    )
    if float(initial[0]) != expected_initial:
        raise RuntimeError("initial raw score does not equal positive-class log-odds")

    return {
        "clf": clf,
        "metadata": metadata,
        "schema": schema,
        "feature_names": feature_names,
        "selected_threshold": selected_threshold,
        "initial_raw_score": float(initial[0]),
        "model_hash": model_hash,
        "metadata_hash": _sha256(artifact_dir / "model_metadata.json"),
        "sklearn_version": sklearn.__version__,
    }


def _render_header(model) -> str:
    clf = model["clf"]
    node_lines: list[str] = []
    tree_lines: list[str] = []
    node_offset = 0

    for estimator in clf.estimators_[:, 0]:
        tree = estimator.tree_
        node_count = int(tree.node_count)
        tree_lines.append(
            f"    QualityTreeRange{{{node_offset}, {node_count}}},"
        )
        for index in range(node_count):
            node_lines.append(
                "    QualityTreeNode{"
                f"{int(tree.children_left[index])}, "
                f"{int(tree.children_right[index])}, "
                f"{int(tree.feature[index])}, "
                f"{_cpp_double(tree.threshold[index])}, "
                f"{_cpp_double(tree.value[index, 0, 0])}"
                "},"
            )
        node_offset += node_count

    feature_lines = [f"    {json.dumps(name)}," for name in model["feature_names"]]
    source_timestamp = model["metadata"]["utc_run_timestamp"]
    return "\n".join([
        "// Generated by tools/python/export/export_quality_model.py.",
        "// Do not edit this file manually.",
        f"// Source artifact run: {source_timestamp}",
        "#pragma once",
        "",
        "#include <array>",
        "#include <cstddef>",
        "#include <string_view>",
        "",
        '#include "edge_audio/quality/quality_model.h"',
        "",
        "namespace edge_audio::quality_model_generated {",
        "",
        f"inline constexpr std::string_view kSchemaVersion = {json.dumps(model['schema']['schema_version'])};",
        f"inline constexpr std::string_view kModelSha256 = {json.dumps(model['model_hash'])};",
        f"inline constexpr std::string_view kMetadataSha256 = {json.dumps(model['metadata_hash'])};",
        f"inline constexpr std::string_view kSklearnVersion = {json.dumps(model['sklearn_version'])};",
        f"inline constexpr std::string_view kSourceRunTimestamp = {json.dumps(source_timestamp)};",
        f"inline constexpr double kLearningRate = {_cpp_double(clf.learning_rate)};",
        f"inline constexpr double kInitialRawScore = {_cpp_double(model['initial_raw_score'])};",
        f"inline constexpr double kSelectedThreshold = {_cpp_double(model['selected_threshold'])};",
        f"inline constexpr std::size_t kTreeCount = {clf.estimators_.shape[0]};",
        f"inline constexpr std::size_t kNodeCount = {node_offset};",
        "",
        "inline constexpr std::array<std::string_view, kQualityFeatureCount> kFeatureNames{{",
        *feature_lines,
        "}};",
        "",
        "inline constexpr std::array<QualityTreeRange, kTreeCount> kTrees{{",
        *tree_lines,
        "}};",
        "",
        "inline constexpr std::array<QualityTreeNode, kNodeCount> kNodes{{",
        *node_lines,
        "}};",
        "",
        "}  // namespace edge_audio::quality_model_generated",
        "",
    ])


def _verify_sklearn_formula(artifact_dir: Path, model) -> None:
    """Check that the exported formula (init + lr * sum(trees), then logistic)
    reproduces scikit-learn exactly on every held-out vector."""
    clf = model["clf"]
    for split in PARITY_SPLITS:
        X, _ = _load_split_features(artifact_dir, split, model["feature_names"])
        raw = np.asarray(clf.decision_function(X), dtype=np.float64)
        manual = np.full(len(X), model["initial_raw_score"], dtype=np.float64)
        for estimator in clf.estimators_[:, 0]:
            manual += clf.learning_rate * estimator.predict(X).reshape(-1)
        if not np.array_equal(raw, manual):
            raise RuntimeError(f"{split}: tree-sum formula differs from decision_function")
        probabilities = np.asarray(clf.predict_proba(X)[:, 1], dtype=np.float64)
        if not np.array_equal(probabilities, 1.0 / (1.0 + np.exp(-raw))):
            raise RuntimeError(f"{split}: logistic link differs from predict_proba")


def _load_split_features(artifact_dir: Path, split: str, feature_names: list[str]):
    with np.load(artifact_dir / f"{split}_features.npz") as table:
        X = np.asarray(table["X"], dtype=np.float32)
        ids = table["stable_example_ids"].astype(str)
        names = table["feature_names"].astype(str).tolist()
    if X.ndim != 2 or X.shape[1] != 27:
        raise RuntimeError(f"unexpected {split} feature shape: {X.shape}")
    if names != feature_names:
        raise RuntimeError(f"{split} feature order differs from model schema")
    if len(set(ids.tolist())) != len(ids):
        raise RuntimeError(f"duplicate stable IDs in {split}")
    if not np.isfinite(X).all():
        raise RuntimeError(f"non-finite feature in {split}")
    return X, ids


def _load_split_probabilities(artifact_dir: Path, split: str, threshold: float):
    """sklearn probabilities recorded at training time, keyed by stable ID."""
    probabilities = {}
    with (artifact_dir / f"{split}_predictions.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            probability = float(row["probability"])  # 17 significant digits: exact
            if int(row["binary_prediction"]) != int(probability >= threshold):
                raise RuntimeError(f"{split}: recorded decision disagrees with probability")
            probabilities[row["stable_example_id"]] = probability
    return probabilities


def render_parity_corpus(artifact_dir: Path) -> bytes:
    metadata = json.loads((artifact_dir / "model_metadata.json").read_text())
    schema = json.loads((artifact_dir / "feature_schema.json").read_text())
    threshold = float(schema["threshold"])
    hashes = metadata["artifact_file_hashes"]
    for split in PARITY_SPLITS:
        for name in (f"{split}_features.npz", f"{split}_predictions.csv"):
            if _sha256(artifact_dir / name) != hashes[name]:
                raise RuntimeError(f"artifact hash mismatch for {name}")

    schema_bytes = schema["schema_version"].encode("ascii")
    if len(schema_bytes) >= 32:
        raise RuntimeError("schema version does not fit the corpus header")
    counts = []
    rows = bytearray()
    for split in PARITY_SPLITS:
        X, ids = _load_split_features(artifact_dir, split, list(schema["feature_names"]))
        probabilities = _load_split_probabilities(artifact_dir, split, threshold)
        if set(probabilities) != set(ids.tolist()):
            raise RuntimeError(f"{split}: features and predictions cover different IDs")
        for example_id, features in zip(ids, X):
            encoded_id = example_id.encode("ascii")
            if len(encoded_id) != 24:
                raise RuntimeError(f"unexpected stable ID: {example_id}")
            probability = probabilities[example_id]
            rows += struct.pack(
                "<24s27fdB", encoded_id, *[float(v) for v in features], probability,
                int(probability >= threshold),
            )
        counts.append(len(X))

    model_hash = hashes["quality_gbt.joblib"].encode("ascii")
    header = struct.pack(
        "<8sIIII32s64sd", CORPUS_MAGIC, CORPUS_VERSION, 27, counts[0], counts[1],
        schema_bytes, model_hash, threshold,
    )
    return header + bytes(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--header", type=Path, default=DEFAULT_HEADER)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    only = parser.add_mutually_exclusive_group()
    only.add_argument("--header-only", action="store_true")
    only.add_argument("--corpus-only", action="store_true")
    args = parser.parse_args()
    artifact_dir = args.artifact_dir.resolve()

    if not args.corpus_only:
        model = _load_authoritative(artifact_dir)
        _verify_sklearn_formula(artifact_dir, model)
        args.header.parent.mkdir(parents=True, exist_ok=True)
        args.header.write_text(_render_header(model))
        nodes = sum(tree.tree_.node_count for tree in model["clf"].estimators_[:, 0])
        print(f"header: {_portable(args.header)} "
              f"({model['clf'].estimators_.shape[0]} trees, {nodes} nodes)")

    if not args.header_only:
        corpus = render_parity_corpus(artifact_dir)
        args.corpus.parent.mkdir(parents=True, exist_ok=True)
        args.corpus.write_bytes(corpus)
        print(f"corpus: {_portable(args.corpus)} ({len(corpus)} bytes)")


if __name__ == "__main__":
    main()
