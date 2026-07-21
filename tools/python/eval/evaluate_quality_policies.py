#!/usr/bin/env python3
"""Compare file-level rule, learned, and heuristic hybrid policies.

This reads only the frozen held-out feature table and saved probabilities. It
does not load sklearn, retrain, mutate artifacts, or run ASR.
"""

from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "quality_model_neura_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, float | int]:
    tp = int(np.sum((targets == 1) & (predictions == 1)))
    fp = int(np.sum((targets == 0) & (predictions == 1)))
    tn = int(np.sum((targets == 0) & (predictions == 0)))
    fn = int(np.sum((targets == 1) & (predictions == 0)))
    far = fp / max(fp + tn, 1)
    frr = fn / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-15)
    calls = int(np.sum(predictions))
    total = int(targets.size)
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "far": far,
        "frr": frr,
        "f1": f1,
        "asr_invocations": calls,
        "asr_calls_avoided": total - calls,
        "asr_calls_avoided_percent": (total - calls) / max(total, 1) * 100.0,
    }


def main() -> int:
    wall_start = time.perf_counter()
    metadata_path = ARTIFACT_DIR / "model_metadata.json"
    table_path = ARTIFACT_DIR / "test_features.npz"
    predictions_path = ARTIFACT_DIR / "test_predictions.csv"
    metadata = json.loads(metadata_path.read_text())

    expected_table_hash = metadata["artifact_file_hashes"][table_path.name]
    if sha256(table_path) != expected_table_hash:
        raise RuntimeError("held-out feature table hash does not match metadata")

    with np.load(table_path, allow_pickle=False) as table:
        targets = table["y"].astype(np.int8)
        rule_predictions = table["gate_predictions"].astype(np.int8)
        example_ids = table["stable_example_ids"].astype(str)
        feature_names = table["feature_names"].astype(str).tolist()

    if feature_names != metadata["ordered_feature_names"]:
        raise RuntimeError("held-out feature order does not match metadata")

    with predictions_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if [row["stable_example_id"] for row in rows] != example_ids.tolist():
        raise RuntimeError("saved predictions do not match held-out example order")

    threshold = float(metadata["selected_threshold"])
    probabilities = np.asarray(
        [float(row["probability"]) for row in rows], dtype=np.float64
    )
    learned_predictions = (probabilities >= threshold).astype(np.int8)
    hybrid_predictions = rule_predictions & learned_predictions

    result = {
        "scope": "full_frozen_held_out_feature_table",
        "examples": int(targets.size),
        "threshold": threshold,
        "schema_version": metadata["ordered_feature_schema_version"],
        "model_sha256": metadata["artifact_file_hashes"]["quality_gbt.joblib"],
        "asr_invocation_definition": (
            "one file-level call per admitted example; ASR itself was not run"
        ),
        "rule_definition": "saved PASS/BORDERLINE file-level gate baseline",
        "hybrid_definition": "saved rule admission AND learned admission",
        "policies": {
            "rule": metrics(targets, rule_predictions),
            "learned": metrics(targets, learned_predictions),
            "hybrid": metrics(targets, hybrid_predictions),
        },
        "decision_table_wall_ms": (time.perf_counter() - wall_start) * 1000.0,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
