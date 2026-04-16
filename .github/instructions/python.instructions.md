---
applyTo: "tools/python/**"
---

Python is offline support tooling only. It is not the runtime.

Rules:
- No runtime Python paths. All inference happens in C++.
- Scripts must have a clear CLI (argparse) and a usage docstring.
- Do not silently ignore unknown JSON fields — warn or skip with a clear message.
- Keep dependencies minimal: stdlib + numpy is acceptable; avoid heavy ML frameworks.
- When C++ logging adds/renames/removes a field, update all Python readers in the same patch.
- simulate_gate() in sweep_thresholds.py must mirror gate.cpp logic exactly.
- Prefer simple flat scripts over class hierarchies for one-off analysis.
- Scripts should be runnable stand-alone: python tools/python/eval/foo.py [args].
