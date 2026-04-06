#!/usr/bin/env python3
"""
bundle.py
Package a pipeline binary + model + optional config into a reproducible
deployment tarball with a manifest.

Output: dist/<bundle_name>.tar.gz  and  dist/<bundle_name>/manifest.json

The manifest records:
  - binary path, size, sha256
  - model path, size, sha256
  - git commit (if available)
  - build timestamp
  - host platform info
  - whisper_version (from vendor/whisper.cpp/CMakeLists.txt if present)
  - CLI flags snapshot (optional --config key=val arguments)

Usage:
    python tools/python/packaging/bundle.py \\
        --binary runtime/cpp/build_opt/audio_pipeline_opt \\
        --model  vendor/whisper.cpp/models/ggml-base.en.bin

    python tools/python/packaging/bundle.py \\
        --binary runtime/cpp/build/audio_pipeline \\
        --model  vendor/whisper.cpp/models/ggml-base.en.bin \\
        --name   pipeline_v1 \\
        --dist   releases/ \\
        --config chunk_ms=5000 threads=4

    python tools/python/packaging/bundle.py --check releases/pipeline_v1.tar.gz
"""

import argparse
import datetime
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from typing import Dict, List, Optional


DIST_DIR = "dist"


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def git_commit(repo_root: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", repo_root, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except Exception:
        return "unknown"


def git_dirty(repo_root: str) -> bool:
    try:
        result = subprocess.run(
            ["git", "-C", repo_root, "status", "--porcelain"],
            capture_output=True, text=True, check=True)
        return bool(result.stdout.strip())
    except Exception:
        return False


def whisper_version(repo_root: str) -> str:
    cmake = os.path.join(repo_root, "vendor", "whisper.cpp", "CMakeLists.txt")
    if not os.path.isfile(cmake):
        return "unknown"
    try:
        with open(cmake) as f:
            content = f.read()
        m = re.search(r'VERSION\s+([\d.]+)', content)
        return m.group(1) if m else "unknown"
    except Exception:
        return "unknown"


def build_manifest(
    binary_path: str,
    model_path: str,
    bundle_name: str,
    config_flags: Dict[str, str],
    repo_root: str,
) -> dict:
    commit = git_commit(repo_root)
    dirty  = git_dirty(repo_root)

    return {
        "bundle_name":     bundle_name,
        "created_at":      datetime.datetime.utcnow().isoformat() + "Z",
        "git_commit":      commit,
        "git_dirty":       dirty,
        "whisper_version": whisper_version(repo_root),
        "platform": {
            "system":   platform.system(),
            "machine":  platform.machine(),
            "python":   platform.python_version(),
            "hostname": platform.node(),
        },
        "binary": {
            "src_path": os.path.abspath(binary_path),
            "filename": os.path.basename(binary_path),
            "size_bytes": os.path.getsize(binary_path),
            "sha256":    sha256_file(binary_path),
        },
        "model": {
            "src_path": os.path.abspath(model_path),
            "filename": os.path.basename(model_path),
            "size_bytes": os.path.getsize(model_path),
            "sha256":    sha256_file(model_path),
        },
        "config": config_flags,
    }


def create_bundle(
    binary_path: str,
    model_path: str,
    bundle_name: str,
    dist_dir: str,
    config_flags: Dict[str, str],
    repo_root: str,
    include_model: bool,
) -> str:
    """
    Assemble files into a temp directory, build manifest, then tar.gz.
    Returns the path to the created tarball.
    """
    manifest = build_manifest(binary_path, model_path, bundle_name,
                               config_flags, repo_root)

    os.makedirs(dist_dir, exist_ok=True)
    tarball_path = os.path.join(dist_dir, f"{bundle_name}.tar.gz")

    with tempfile.TemporaryDirectory() as tmpdir:
        bundle_dir = os.path.join(tmpdir, bundle_name)
        os.makedirs(bundle_dir)

        # Copy binary and make executable
        bin_dst = os.path.join(bundle_dir, os.path.basename(binary_path))
        shutil.copy2(binary_path, bin_dst)
        os.chmod(bin_dst, 0o755)

        # Copy model (optional — can be large; skip with --no-model)
        if include_model:
            model_dst = os.path.join(bundle_dir, os.path.basename(model_path))
            shutil.copy2(model_path, model_dst)

        # Write manifest
        manifest_path = os.path.join(bundle_dir, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
            f.write("\n")

        # Write a minimal run.sh inside the bundle
        run_sh = os.path.join(bundle_dir, "run.sh")
        _write_run_sh(run_sh, binary_path, model_path, include_model, config_flags)
        os.chmod(run_sh, 0o755)

        # Create tarball
        with tarfile.open(tarball_path, "w:gz") as tar:
            tar.add(bundle_dir, arcname=bundle_name)

    return tarball_path


def _write_run_sh(
    path: str,
    binary_path: str,
    model_path: str,
    model_bundled: bool,
    config_flags: Dict[str, str],
):
    bin_name   = os.path.basename(binary_path)
    model_name = os.path.basename(model_path)
    model_ref  = f'"./{model_name}"' if model_bundled else f'"$MODEL_PATH"'

    extra_args = " \\\n    ".join(
        f'--{k.replace("_", "-")} {v}' for k, v in config_flags.items()
    )
    extra_block = f"    {extra_args} \\\n" if extra_args else ""

    content = f"""#!/usr/bin/env bash
# Minimal run script bundled with {bin_name}
# Usage: bash run.sh <input.wav> [extra args]
set -euo pipefail
DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
BIN="$DIR/{bin_name}"
MODEL={model_ref}
INPUT="${{1:?Usage: run.sh <input.wav>}}"
shift || true

"$BIN" \\
    --input "$INPUT" \\
    --model "$MODEL" \\
{extra_block}    "$@"
"""
    with open(path, "w") as f:
        f.write(content)


def check_bundle(tarball_path: str):
    """Verify a bundle: extract manifest, check sha256 hashes of included files."""
    print(f"Checking: {tarball_path}")

    with tarfile.open(tarball_path, "r:gz") as tar:
        # Find manifest.json
        manifest_member = next(
            (m for m in tar.getmembers() if m.name.endswith("manifest.json")), None)
        if manifest_member is None:
            print("ERROR: manifest.json not found in archive.", file=sys.stderr)
            sys.exit(1)
        f = tar.extractfile(manifest_member)
        manifest = json.load(f)

        print(f"  bundle   : {manifest.get('bundle_name', '?')}")
        print(f"  created  : {manifest.get('created_at', '?')}")
        print(f"  commit   : {manifest.get('git_commit', '?')}"
              f"{'  (dirty)' if manifest.get('git_dirty') else ''}")
        print(f"  platform : {manifest.get('platform', {}).get('system', '?')} "
              f"{manifest.get('platform', {}).get('machine', '?')}")
        print(f"  whisper  : {manifest.get('whisper_version', '?')}")

        # Check files present in archive and verify hashes
        members = {m.name: m for m in tar.getmembers()}
        bundle_name = manifest.get("bundle_name", "")
        ok = True

        for role, info in [("binary", manifest["binary"]), ("model", manifest["model"])]:
            fname = info["filename"]
            arc_path = f"{bundle_name}/{fname}"
            if arc_path not in members:
                print(f"  {role:8} {fname}  NOT IN ARCHIVE (model may have been excluded)")
                continue
            fobj = tar.extractfile(members[arc_path])
            h = hashlib.sha256()
            while chunk := fobj.read(1 << 20):
                h.update(chunk)
            digest = h.hexdigest()
            expected = info["sha256"]
            match = digest == expected
            status = "OK" if match else "MISMATCH"
            print(f"  {role:8} {fname}  sha256={digest[:16]}...  {status}")
            if not match:
                ok = False

    if ok:
        print("Bundle integrity: PASS")
    else:
        print("Bundle integrity: FAIL — sha256 mismatch", file=sys.stderr)
        sys.exit(1)


def parse_config_flags(args_list: List[str]) -> Dict[str, str]:
    result = {}
    for item in args_list:
        if "=" in item:
            k, _, v = item.partition("=")
            result[k.strip()] = v.strip()
        else:
            print(f"WARNING: config item ignored (expected key=val): {item}",
                  file=sys.stderr)
    return result


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--binary", default="",
                        help="Path to compiled audio_pipeline binary")
    parser.add_argument("--model", default="",
                        help="Path to ggml model .bin file")
    parser.add_argument("--name", default="",
                        help="Bundle name (default: <binary_stem>__<model_stem>)")
    parser.add_argument("--dist", default=DIST_DIR,
                        help=f"Output directory for tarball (default: {DIST_DIR}/)")
    parser.add_argument("--config", nargs="*", default=[],
                        metavar="KEY=VAL",
                        help="Pipeline config flags to embed in manifest and run.sh "
                             "(e.g. chunk_ms=5000 threads=4)")
    parser.add_argument("--no-model", action="store_true",
                        help="Exclude model from tarball (for very large models); "
                             "only manifest is written")
    parser.add_argument("--check", default="",
                        metavar="TARBALL",
                        help="Verify an existing bundle instead of creating one")

    args = parser.parse_args()

    # --- check mode ---
    if args.check:
        if not os.path.isfile(args.check):
            print(f"ERROR: not found: {args.check}", file=sys.stderr)
            sys.exit(1)
        check_bundle(args.check)
        return

    # --- create mode ---
    if not args.binary or not args.model:
        parser.print_help()
        print("\nERROR: --binary and --model are required.", file=sys.stderr)
        sys.exit(1)

    for p, label in [(args.binary, "--binary"), (args.model, "--model")]:
        if not os.path.isfile(p):
            print(f"ERROR: {label} not found: {p}", file=sys.stderr)
            sys.exit(1)

    config_flags = parse_config_flags(args.config or [])

    bundle_name = args.name or (
        f"{os.path.splitext(os.path.basename(args.binary))[0]}"
        f"__{os.path.splitext(os.path.basename(args.model))[0]}"
    )

    # Repo root: two levels up from this script (tools/python/packaging/)
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", ".."))

    print(f"Bundling:")
    print(f"  binary : {args.binary}")
    print(f"  model  : {args.model}")
    print(f"  name   : {bundle_name}")
    print(f"  dist   : {args.dist}/")
    if config_flags:
        print(f"  config : {config_flags}")
    if args.no_model:
        print(f"  NOTE   : model excluded from tarball (--no-model)")

    tarball = create_bundle(
        binary_path=args.binary,
        model_path=args.model,
        bundle_name=bundle_name,
        dist_dir=args.dist,
        config_flags=config_flags,
        repo_root=repo_root,
        include_model=not args.no_model,
    )

    size_mb = os.path.getsize(tarball) / (1 << 20)
    print(f"\nCreated: {tarball}  ({size_mb:.1f} MB)")
    print(f"Verify:  python tools/python/packaging/bundle.py --check {tarball}")


if __name__ == "__main__":
    main()
