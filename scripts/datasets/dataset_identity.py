"""Stable dataset identities, deterministic partitions, and overlap checks."""

from __future__ import annotations

import hashlib
import json
import os
import random
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Callable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]


def _normalized_absolute(path: str | Path, repo_root: Path) -> Path:
    value = Path(path)
    if not value.is_absolute():
        value = repo_root / value
    return Path(os.path.normpath(str(value)))


@lru_cache(maxsize=16)
def _cached_root_mappings(repo_root_text: str) -> tuple[tuple[Path, Path], ...]:
    """Return physical-root to stable logical-root mappings."""
    repo_root = Path(repo_root_text)
    mappings: list[tuple[Path, Path]] = []
    raw_dir = repo_root / "data" / "raw"
    if raw_dir.exists():
        for child in sorted(raw_dir.iterdir(), key=lambda p: p.name):
            mappings.append((child.resolve(strict=False), Path("data/raw") / child.name))
    mappings.append((repo_root.resolve(strict=False), Path()))
    mappings.sort(key=lambda item: len(item[0].parts), reverse=True)
    return tuple(mappings)


def _root_mappings(repo_root: Path) -> tuple[tuple[Path, Path], ...]:
    return _cached_root_mappings(str(repo_root.resolve(strict=False)))


def canonical_source_identity(
    path: str | Path,
    repo_root: Path = REPO_ROOT,
) -> str:
    """Map relative, absolute, and symlink-resolved paths to one stable identity."""
    if path is None or str(path) == "":
        return ""

    repo_root = Path(repo_root).resolve(strict=False)
    absolute = _normalized_absolute(path, repo_root)

    # Most manifest records are already lexical paths through the repository's
    # declared data/raw roots. Preserve that stable spelling without a syscall.
    try:
        lexical = absolute.relative_to(repo_root)
    except ValueError:
        lexical = None
    if lexical is not None and lexical.parts[:2] == ("data", "raw"):
        return lexical.as_posix()

    resolved = absolute.resolve(strict=False)

    for physical_root, logical_root in _root_mappings(repo_root):
        try:
            suffix = resolved.relative_to(physical_root)
        except ValueError:
            continue
        logical = logical_root / suffix
        return logical.as_posix()

    raise ValueError(f"path is outside the repository and declared dataset roots: {path}")


def portable_repo_path(path: str | Path, repo_root: Path = REPO_ROOT) -> str:
    """Return a portable repository/dataset-relative path without host prefixes."""
    if path is None or str(path) == "":
        return ""

    repo_root = Path(repo_root).resolve(strict=False)
    absolute = _normalized_absolute(path, repo_root)
    try:
        return absolute.relative_to(repo_root).as_posix()
    except ValueError:
        return canonical_source_identity(absolute, repo_root)


def is_path_reference(value: object) -> bool:
    if not isinstance(value, str) or not value:
        return False
    return Path(value).is_absolute() or value.startswith("data/") or "/" in value


def portable_reference(value: object, repo_root: Path = REPO_ROOT) -> str:
    """Normalize path-like values while preserving literal transformation strings."""
    if value is None or value == "":
        return ""
    text = str(value)
    return portable_repo_path(text, repo_root) if is_path_reference(text) else text


def input_source_identities(record: Mapping, repo_root: Path = REPO_ROOT) -> set[str]:
    """Return every raw input identity used to construct a labeled example."""
    identities: set[str] = set()
    source = record.get("source", "")
    if source:
        identities.add(canonical_source_identity(str(source), repo_root))
    corruption = record.get("corruption_source", "")
    if is_path_reference(corruption):
        identities.add(canonical_source_identity(str(corruption), repo_root))
    return identities


def deterministic_example_payload(record: Mapping, repo_root: Path = REPO_ROOT) -> dict:
    """Return the transformation-complete, path-stable example description."""
    source = record.get("source", "")
    corruption = record.get("corruption_source", "")
    return {
        "label": record.get("label", ""),
        "source": canonical_source_identity(str(source), repo_root) if source else "",
        "corruption_source": (
            canonical_source_identity(str(corruption), repo_root)
            if is_path_reference(corruption)
            else str(corruption or "")
        ),
        "snr_db": record.get("snr_db"),
        "rir_id": record.get("rir_id", ""),
        "generation_params": record.get("generation_params", {}),
        "duration_sec": record.get("duration_sec"),
        "sample_rate": record.get("sample_rate"),
    }


def deterministic_example_identity(record: Mapping, repo_root: Path = REPO_ROOT) -> str:
    payload = deterministic_example_payload(record, repo_root)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def stable_example_id(record: Mapping, repo_root: Path = REPO_ROOT) -> str:
    return deterministic_example_identity(record, repo_root)[:24]


def deterministic_partition(
    records: Sequence[Mapping],
    split_names: Sequence[str],
    seed: int,
    identity_fn: Callable[[Mapping], str],
) -> dict[str, list[Mapping]]:
    """Deduplicate, shuffle deterministically, and partition records round-robin."""
    if not split_names:
        raise ValueError("split_names must not be empty")

    unique: dict[str, Mapping] = {}
    for record in records:
        identity = identity_fn(record)
        if not identity:
            raise ValueError("empty identity in deterministic partition")
        unique.setdefault(identity, record)

    ordered = [unique[key] for key in sorted(unique)]
    random.Random(seed).shuffle(ordered)
    result: dict[str, list[Mapping]] = {name: [] for name in split_names}
    for index, record in enumerate(ordered):
        result[split_names[index % len(split_names)]].append(record)
    return result


def load_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with open(path) as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cheap_content_hash(
    path: str | Path,
    repo_root: Path = REPO_ROOT,
    max_bytes: int = 2 * 1024 * 1024,
) -> str | None:
    """Hash small rendered files, but do not follow clean-speech symlinks."""
    value = _normalized_absolute(path, Path(repo_root).resolve(strict=False))
    if value.is_symlink() or not value.is_file() or value.stat().st_size > max_bytes:
        return None
    return sha256_file(value)


def split_summary(records: Sequence[Mapping]) -> dict:
    target_counts = Counter(str(row.get("should_transcribe", "")) for row in records)
    label_counts = Counter(str(row.get("label", "")) for row in records)
    return {
        "rows": len(records),
        "class_counts": dict(sorted(target_counts.items())),
        "label_counts": dict(sorted(label_counts.items())),
        "unique_output_paths": len({str(row.get("path", "")) for row in records}),
        "unique_base_utterance_ids": len({
            str(row.get("base_utterance_id"))
            for row in records if row.get("base_utterance_id")
        }),
    }


def _content_hashes(records: Sequence[Mapping], repo_root: Path) -> set[str]:
    hashes = set()
    for record in records:
        value = cheap_content_hash(str(record.get("path", "")), repo_root)
        if value:
            hashes.add(value)
    return hashes


def build_overlap_report(
    splits: Mapping[str, Sequence[Mapping]],
    repo_root: Path = REPO_ROOT,
    include_content_hashes: bool = False,
) -> dict:
    """Compute all mandatory pairwise overlap dimensions."""
    repo_root = Path(repo_root).resolve(strict=False)
    names = list(splits)
    indexed: dict[str, dict[str, set[str]]] = {}

    for name, records in splits.items():
        source_ids: set[str] = set()
        for record in records:
            source_ids.update(input_source_identities(record, repo_root))
        indexed[name] = {
            "canonical_source_identity": source_ids,
            "base_utterance_id": {
                str(record.get("base_utterance_id"))
                for record in records if record.get("base_utterance_id")
            },
            "derived_fingerprint": {
                deterministic_example_identity(record, repo_root) for record in records
            },
            "output_path": {
                portable_repo_path(str(record.get("path", "")), repo_root)
                for record in records if record.get("path")
            },
            "content_hash": _content_hashes(records, repo_root) if include_content_hashes else set(),
        }

    pairs: dict[str, dict] = {}
    forbidden_total = 0
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            dimensions = {}
            for dimension in (
                "canonical_source_identity",
                "base_utterance_id",
                "derived_fingerprint",
                "output_path",
                "content_hash",
            ):
                overlap = sorted(indexed[left][dimension] & indexed[right][dimension])
                dimensions[dimension] = {"count": len(overlap), "examples": overlap[:5]}
                forbidden_total += len(overlap)
            pairs[f"{left}__{right}"] = dimensions

    return {
        "schema_version": "quality-split-overlap-v1",
        "splits": {name: split_summary(records) for name, records in splits.items()},
        "pairs": pairs,
        "include_content_hashes": include_content_hashes,
        "forbidden_overlap_total": forbidden_total,
        "passed": forbidden_total == 0,
    }


def assert_no_forbidden_overlap(report: Mapping) -> None:
    if not report.get("passed", False):
        raise RuntimeError(
            f"forbidden cross-split overlap detected: {report.get('forbidden_overlap_total')}"
        )


def print_overlap_matrix(report: Mapping) -> None:
    print("pair                         source  base_id  derived  path  content")
    for pair, dimensions in report.get("pairs", {}).items():
        print(
            f"{pair:28s}"
            f" {dimensions['canonical_source_identity']['count']:6d}"
            f" {dimensions['base_utterance_id']['count']:8d}"
            f" {dimensions['derived_fingerprint']['count']:8d}"
            f" {dimensions['output_path']['count']:5d}"
            f" {dimensions['content_hash']['count']:8d}"
        )
