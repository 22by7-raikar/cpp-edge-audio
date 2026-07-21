#!/usr/bin/env python3
"""Export the authoritative sklearn quality GBT and its parity corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.dummy import DummyClassifier


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ARTIFACT_DIR = REPO_ROOT / "artifacts/quality_model_neura_v1"
DEFAULT_HEADER = REPO_ROOT / "runtime/cpp/generated/quality_model_neura_v1.h"
DEFAULT_CORPUS = (
    REPO_ROOT / "runtime/cpp/tests/data/quality_model_neura_v1_parity.bin"
)
CORPUS_MAGIC = b"QMPAR1\0\0"
CORPUS_VERSION = 1


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
        '#include "gate/quality_model.h"',
        "",
        "namespace pipeline::quality_model_generated {",
        "",
        f"inline constexpr std::string_view kSchemaVersion = {json.dumps(model['schema']['schema_version'])};",
        f"inline constexpr std::string_view kModelSha256 = {json.dumps(model['model_hash'])};",
        f"inline constexpr std::string_view kMetadataSha256 = {json.dumps(model['metadata_hash'])};",
        f"inline constexpr std::string_view kSklearnVersion = {json.dumps(sklearn.__version__)};",
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
        "}  // namespace pipeline::quality_model_generated",
        "",
    ])


def _load_parity_split(artifact_dir: Path, split: str, feature_names: list[str]):
    with np.load(artifact_dir / f"{split}_features.npz") as table:
        X = np.asarray(table["X"], dtype=np.float32)
        ids = table["stable_example_ids"].astype(str)
        names = table["feature_names"].astype(str).tolist()
    if X.shape != (1700, 27):
        raise RuntimeError(f"unexpected {split} feature shape: {X.shape}")
    if names != feature_names:
        raise RuntimeError(f"{split} feature order differs from model schema")
    if len(set(ids.tolist())) != len(ids):
        raise RuntimeError(f"duplicate stable IDs in {split}")
    if not np.isfinite(X).all():
        raise RuntimeError(f"non-finite feature in {split}")
    return X, ids


def _render_corpus(artifact_dir: Path, model) -> bytes:
    clf = model["clf"]
    validation_X, validation_ids = _load_parity_split(
        artifact_dir, "validation", model["feature_names"]
    )
    test_X, test_ids = _load_parity_split(
        artifact_dir, "test", model["feature_names"]
    )
    X = np.concatenate([validation_X, test_X], axis=0)
    ids = np.concatenate([validation_ids, test_ids], axis=0)
    if len(set(ids.tolist())) != len(ids):
        raise RuntimeError("stable IDs overlap across validation and test")

    raw = np.asarray(clf.decision_function(X), dtype=np.float64)
    probabilities = np.asarray(clf.predict_proba(X)[:, 1], dtype=np.float64)
    manual = np.full(len(X), model["initial_raw_score"], dtype=np.float64)
    for estimator in clf.estimators_[:, 0]:
        manual += clf.learning_rate * estimator.predict(X).reshape(-1)
    if not np.array_equal(raw, manual):
        difference = float(np.max(np.abs(raw - manual)))
        raise RuntimeError(f"manual raw-score formula differs by {difference}")
    logistic = 1.0 / (1.0 + np.exp(-raw))
    if not np.array_equal(probabilities, logistic):
        difference = float(np.max(np.abs(probabilities - logistic)))
        raise RuntimeError(f"logistic probability formula differs by {difference}")

    schema_bytes = model["schema"]["schema_version"].encode("ascii")
    if len(schema_bytes) >= 32:
        raise RuntimeError("schema version does not fit corpus header")
    output = bytearray(struct.pack(
        "<8sIIII32s64s",
        CORPUS_MAGIC,
        CORPUS_VERSION,
        X.shape[1],
        len(validation_X),
        len(test_X),
        schema_bytes,
        model["model_hash"].encode("ascii"),
    ))
    row_format = "<24s27fddB"
    for example_id, features, score, probability in zip(
        ids, X, raw, probabilities
    ):
        encoded_id = example_id.encode("ascii")
        if len(encoded_id) != 24:
            raise RuntimeError(f"unexpected stable ID: {example_id}")
        decision = int(probability >= model["selected_threshold"])
        output.extend(struct.pack(
            row_format,
            encoded_id,
            *[float(value) for value in features],
            float(score),
            float(probability),
            decision,
        ))
    return bytes(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--header", type=Path, default=DEFAULT_HEADER)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    args = parser.parse_args()

    artifact_dir = args.artifact_dir.resolve()
    model = _load_authoritative(artifact_dir)
    header = _render_header(model)
    corpus = _render_corpus(artifact_dir, model)

    print("Planned generated outputs:")
    print(f"  header: {_portable(args.header)}")
    print(f"  corpus: {_portable(args.corpus)}")
    args.header.parent.mkdir(parents=True, exist_ok=True)
    args.corpus.parent.mkdir(parents=True, exist_ok=True)
    args.header.write_text(header)
    args.corpus.write_bytes(corpus)
    print(f"Model SHA-256: {model['model_hash']}")
    print(f"Trees: {model['clf'].estimators_.shape[0]}")
    print(f"Nodes: {sum(tree.tree_.node_count for tree in model['clf'].estimators_[:, 0])}")
    print(f"Parity examples: 3400")


if __name__ == "__main__":
    main()
