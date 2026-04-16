---
applyTo: "runtime/cpp/**"
---

C++17. CMake. Linux-first. No heavy external libraries beyond whisper.cpp.

Rules:
- Read existing headers before editing. Do not invent APIs.
- One module change per patch. Do not refactor unrelated code in the same change.
- No speculative abstractions. No helper classes for one-off operations.
- Maintain memory safety and correct struct alignment.
- When modifying gate/chunker/scene/adaptive/logger, add a test in the same patch.
- Gate logic must remain interpretable and rule-based. No ML in the gate.
- GateConfig defaults must stay conservative (prefer BORDERLINE over false FAILs).
- Reason strings are part of the stable log schema — do not rename or remove them.
- New gate thresholds must also be wired into tools/python/tuning/sweep_thresholds.py simulate_gate().
- CLI flags in main.cpp should only expose the most important knobs.
  Do not add a flag for every threshold — keep defaults in code sensible.
