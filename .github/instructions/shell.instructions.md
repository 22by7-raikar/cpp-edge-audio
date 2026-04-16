---
applyTo: "scripts/**,benchmarks/**"
---

Shell scripts are Linux-first. Use bash with set -euo pipefail.

Rules:
- Always set: set -euo pipefail at the top.
- Use $(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd) to get REPO_ROOT reliably.
- Quote all variable expansions: "$var", not $var.
- Check for required binaries/files early and fail with a clear error message.
- Do not use bashisms that break on older bash 4.x (no associative array declare -A if avoidable).
- Do not hardcode absolute paths. Derive from REPO_ROOT.
- Print a short summary of what the script will do before running.
- Write outputs to named subdirectories under benchmarks/results/ or data/.
- Do not silently overwrite results — use timestamped or labeled output names.
