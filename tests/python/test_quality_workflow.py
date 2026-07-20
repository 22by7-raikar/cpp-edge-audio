import json
import shlex
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "datasets"))
sys.path.insert(0, str(REPO_ROOT / "tools" / "python" / "training"))

from dataset_identity import (  # noqa: E402
    assert_no_forbidden_overlap,
    build_overlap_report,
    canonical_source_identity,
    deterministic_example_identity,
    deterministic_partition,
)
from train_quality_model import (  # noqa: E402
    FEATURE_NAMES,
    _artifact_hashes,
    _assert_metrics_document_consistent,
    _build_model_metadata,
    _legacy_metrics_payload,
    _metrics_document,
)


class DatasetIdentityTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name) / "repo"
        self.external = Path(self.tempdir.name) / "physical_audio"
        self.root.mkdir()
        self.external.mkdir()
        raw = self.root / "data" / "raw"
        raw.mkdir(parents=True)
        (raw / "sample_set").symlink_to(self.external, target_is_directory=True)
        self.audio = self.external / "clip.wav"
        self.audio.write_bytes(b"RIFF-test")

    def tearDown(self):
        self.tempdir.cleanup()

    def test_absolute_and_relative_canonical_source_equivalence(self):
        relative = "data/raw/sample_set/clip.wav"
        absolute = self.root / relative
        self.assertEqual(
            canonical_source_identity(relative, self.root),
            canonical_source_identity(absolute, self.root),
        )

    def test_symlink_and_resolved_source_equivalence(self):
        lexical = self.root / "data" / "raw" / "sample_set" / "clip.wav"
        self.assertEqual(
            canonical_source_identity(lexical, self.root),
            canonical_source_identity(self.audio.resolve(), self.root),
        )

    def test_fingerprint_is_path_stable_and_transformation_complete(self):
        base = {
            "label": "clipped_or_distorted",
            "source": "data/raw/sample_set/clip.wav",
            "corruption_source": "hard_clip@0.20",
            "snr_db": None,
            "rir_id": "",
            "generation_params": {"clip_threshold": 0.2},
            "duration_sec": 5.0,
            "sample_rate": 16000,
        }
        absolute = dict(base, source=str(self.root / base["source"]))
        changed = dict(base, generation_params={"clip_threshold": 0.3})
        self.assertEqual(
            deterministic_example_identity(base, self.root),
            deterministic_example_identity(absolute, self.root),
        )
        self.assertNotEqual(
            deterministic_example_identity(base, self.root),
            deterministic_example_identity(changed, self.root),
        )
        changed_snr = dict(base, snr_db=3.125)
        self.assertNotEqual(
            deterministic_example_identity(base, self.root),
            deterministic_example_identity(changed_snr, self.root),
        )

    def test_cross_split_duplicate_detection_and_abort(self):
        shared = "data/raw/sample_set/clip.wav"
        record = {
            "path": "data/processed/quality_train/a.wav",
            "label": "clean_speech",
            "should_transcribe": "yes",
            "source": shared,
            "base_utterance_id": "utterance-1",
            "corruption_source": "",
            "generation_params": {},
        }
        other = dict(record, path="data/processed/quality_val/a.wav")
        report = build_overlap_report(
            {"train": [record], "validation": [other]}, self.root,
        )
        self.assertFalse(report["passed"])
        self.assertEqual(
            report["pairs"]["train__validation"]
            ["canonical_source_identity"]["count"],
            1,
        )
        with self.assertRaises(RuntimeError):
            assert_no_forbidden_overlap(report)

    def test_deterministic_partition(self):
        records = [{"id": str(index)} for index in range(11)]
        kwargs = {
            "records": records,
            "split_names": ["train", "validation", "test"],
            "seed": 17,
            "identity_fn": lambda row: row["id"],
        }
        first = deterministic_partition(**kwargs)
        second = deterministic_partition(**kwargs)
        self.assertEqual(first, second)
        flattened = [row["id"] for rows in first.values() for row in rows]
        self.assertCountEqual(flattened, [row["id"] for row in records])
        self.assertEqual(len(flattened), len(set(flattened)))


class TrainingArtifactTest(unittest.TestCase):
    def setUp(self):
        self.entries = [
            {"label": "clean_speech", "should_transcribe": "yes"},
            {"label": "music", "should_transcribe": "no"},
        ]
        self.y_true = np.asarray([1, 0])
        self.probabilities = np.asarray([0.45, 0.45])

    def test_metrics_json_uses_selected_threshold(self):
        document = _metrics_document(
            "validation", 0.4, self.y_true, self.probabilities, self.entries,
        )
        with tempfile.TemporaryDirectory() as tempdir:
            metrics_path = Path(tempdir) / "validation_metrics.json"
            metrics_path.write_text(json.dumps(document))
            saved = json.loads(metrics_path.read_text())

        selected = saved["metrics_at_selected_threshold"]
        default = saved["default_threshold_0_5_diagnostic"]["metrics"]
        self.assertEqual((selected["tp"], selected["fp"]), (1, 1))
        self.assertEqual((default["fn"], default["tn"]), (1, 1))
        _assert_metrics_document_consistent(
            saved, 0.4, self.y_true, self.probabilities, self.entries,
        )

        saved["selected_threshold"] = 0.5
        with self.assertRaises(RuntimeError):
            _assert_metrics_document_consistent(
                saved, 0.4, self.y_true, self.probabilities, self.entries,
            )

    def test_legacy_metrics_json_threshold_is_consistent(self):
        payload = _legacy_metrics_payload(
            "run-id", 0.4, self.y_true, self.probabilities, self.entries,
        )
        self.assertEqual(payload["threshold"], 0.4)
        self.assertEqual((payload["metrics"]["tp"], payload["metrics"]["fp"]), (1, 1))
        diagnostic = payload["default_threshold_0_5_diagnostic"]
        self.assertEqual(diagnostic["threshold"], 0.5)
        self.assertEqual(
            (diagnostic["metrics"]["fn"], diagnostic["metrics"]["tn"]),
            (1, 1),
        )

    def test_metadata_and_artifact_hash_generation(self):
        class FakeModel:
            def get_params(self):
                return {"n_estimators": 100, "random_state": 42}

        with tempfile.TemporaryDirectory() as tempdir:
            artifact_dir = Path(tempdir)
            (artifact_dir / "quality_gbt.joblib").write_bytes(b"model")
            (artifact_dir / "validation_metrics.json").write_text("{}\n")
            hashes = _artifact_hashes(artifact_dir)
            self.assertEqual(
                set(hashes), {"quality_gbt.joblib", "validation_metrics.json"},
            )
            self.assertTrue(all(len(value) == 64 for value in hashes.values()))

            paths = {
                name: REPO_ROOT / "data" / "labels" / f"quality_{name}.jsonl"
                for name in ("train", "validation", "test")
            }
            records = {
                name: [{
                    "path": f"data/processed/{name}/a.wav",
                    "label": "clean_speech",
                    "should_transcribe": "yes",
                    "base_utterance_id": f"{name}-1",
                }]
                for name in paths
            }
            metadata = _build_model_metadata(
                FakeModel(),
                0.4,
                paths,
                {name: "a" * 64 for name in paths},
                records,
                {"commit": "abc", "dirty": True},
                artifact_dir,
                [
                    sys.executable,
                    str(
                        REPO_ROOT
                        / "tools/python/training/train_quality_model.py"
                    ),
                    "--authoritative-protocol",
                    "--train-labels",
                    str(REPO_ROOT / "data/labels/quality_train.jsonl"),
                    "--seed",
                    "42",
                ],
            )
            expected_argv = [
                "python",
                "tools/python/training/train_quality_model.py",
                "--authoritative-protocol",
                "--train-labels",
                "data/labels/quality_train.jsonl",
                "--seed",
                "42",
            ]
            self.assertEqual(metadata["selected_threshold"], 0.4)
            self.assertEqual(metadata["ordered_feature_names"], FEATURE_NAMES)
            self.assertEqual(metadata["artifact_file_hashes"], hashes)
            self.assertEqual(metadata["argv"], expected_argv)
            self.assertEqual(
                metadata["exact_command_line"], shlex.join(expected_argv),
            )
            self.assertNotIn("/home/", metadata["exact_command_line"])
            self.assertNotIn("pytest", metadata["exact_command_line"])
            self.assertNotIn("__main__.py", metadata["exact_command_line"])
            self.assertEqual(metadata["argv"][0], "python")
            self.assertEqual(
                metadata["label_files"]["validation"]["summary"]["rows"], 1,
            )


if __name__ == "__main__":
    unittest.main()
