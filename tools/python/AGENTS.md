# Python agent rules

Scope:
Python in this repo is for support tooling only.

Use Python for:
- offline evaluation
- plotting
- threshold tuning
- quick utilities
- data preparation
- log analysis
- simple prototyping before C++ porting when explicitly requested

Environment:
- Use tools/python/edge_stt
- Do not assume .venv
- Do not assume conda

Preferred packages:
- numpy
- scipy
- pandas
- librosa
- matplotlib
- soundfile
- sounddevice
- jiwer
- onnxruntime only if explicitly needed

Rules:
- Do not turn Python into the main runtime
- Keep scripts CLI-runnable
- Never use emojis
- Do not create extra markdown docs unless explicitly requested
