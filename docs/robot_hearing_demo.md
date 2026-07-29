# Robot Hearing presentation runbook

The primary presentation path is deterministic base.en replay on CUDA with rule
policy. The live iPhone path is optional and experimental.

## Prerequisites and variables

CUDA builds require CMake 3.18 or newer, a working CUDA toolkit, FFmpeg, Python
3, and initialized whisper.cpp sources. Keep recordings, models, and results
outside version control.

```bash
export REPO="$(git rev-parse --show-toplevel)"
export DEMO="$REPO/runtime/cpp/build/robot_hearing_demo"
export BASE_MODEL="$REPO/vendor/whisper.cpp/models/ggml-base.en.bin"
export TINY_MODEL="$REPO/vendor/whisper.cpp/models/ggml-tiny.en.bin"
export RULE_FIXTURE="/path/to/private/rule-command-sequence.wav"
export LEARNED_FIXTURE="/path/to/private/learned-command-sequence.wav"
export RESULTS_DIR="/path/to/private/robot-hearing-results"
```

## Preflight

```bash
cd "$REPO"
git branch --show-current
git status --short
test -x "$DEMO"
test -f "$BASE_MODEL"
test -f "$TINY_MODEL"
test -f "$RULE_FIXTURE"
test -f "$LEARNED_FIXTURE"
mkdir -p "$RESULTS_DIR"
```

Configure and build the presentation binary when needed:

```bash
cmake -S "$REPO/runtime/cpp" -B "$REPO/runtime/cpp/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DWHISPER_ROOT="$REPO/vendor/whisper.cpp" \
  -DGGML_CUDA=ON \
  -DBUILD_TESTS=ON
cmake --build "$REPO/runtime/cpp/build" -j"$(nproc)"
(cd "$REPO/runtime/cpp/build" && ctest --output-on-failure)
```

Run CUDA work from the ordinary Linux terminal. A restricted development
sandbox may not expose the NVIDIA device.

## Primary deterministic replay

```bash
CUDA_VISIBLE_DEVICES=0 \
"$REPO/scripts/benchmark_robot_hearing_replay.sh" \
  --input "$RULE_FIXTURE" \
  --demo "$DEMO" \
  --model "$BASE_MODEL" \
  --model-label base.en \
  --quality-policy rule \
  --capture-device deterministic_replay \
  --output-dir "$RESULTS_DIR"
```

The script prints the exact result directory and writes:

- `stdout.jsonl` for robot events;
- `stderr.log` for runner, Whisper, and backend diagnostics; and
- `invocation.txt` for the effective replay inputs.

It refuses to replace an existing deterministic result unless `--force` is
explicitly supplied. Prefer a fresh results directory for presentation.

## Verify JSONL

Set `RULE_RESULT` to the `result_dir` printed by the runner:

```bash
export RULE_RESULT="$RESULTS_DIR/<fixture-stem>_base.en_rule"

python3 - "$RULE_RESULT/stdout.jsonl" <<'PY'
import json
import sys

required = {
    "event", "utterance_id", "duration_ms", "quality_policy", "admitted",
    "asr_ran", "transcript", "intent", "action", "latency", "stream",
}
events = []
with open(sys.argv[1], encoding="utf-8") as handle:
    for line_number, line in enumerate(handle, 1):
        if not line.strip():
            continue
        event = json.loads(line)
        missing = required.difference(event)
        if missing:
            raise SystemExit(f"line {line_number}: missing {sorted(missing)}")
        events.append(event)

print(f"valid_events={len(events)}")
for event in events:
    print(
        event["utterance_id"],
        repr(event["transcript"]),
        event["intent"],
        event["action"],
    )
PY
```

## Verify CUDA

```bash
grep -E \
  'gpu_device = 0|using CUDA0 backend|backend_active=cuda|cpu_fallback=no' \
  "$RULE_RESULT/stderr.log"
```

Do not report sandbox CPU timing as deployment GPU latency.

## Fields to explain

| Field | Meaning |
|---|---|
| `utterance_id` | Monotonic identifier for one finalized utterance |
| `duration_ms` | Finalized utterance audio duration, including retained context |
| `quality_policy` | `rule`, `learned`, or `hybrid` |
| `admitted` | Whether this completed utterance may invoke ASR |
| `asr_ran` | Whether Whisper actually ran |
| `transcript` | Whisper output when ASR succeeds |
| `intent` | Conservative router result |
| `action` | Robot action string, or null for `UNKNOWN` |
| `quality_us` | Completed-utterance quality-analysis time |
| `asr_ms` | Whisper wrapper inference time |
| `post_utterance_ms` | Processing after utterance finalization, not capture-to-action latency |

## Optional learned-policy replay

The learned demonstration uses the same base model and frozen threshold `0.3`;
the replay helper supplies that threshold to the executable.

```bash
CUDA_VISIBLE_DEVICES=0 \
"$REPO/scripts/benchmark_robot_hearing_replay.sh" \
  --input "$LEARNED_FIXTURE" \
  --demo "$DEMO" \
  --model "$BASE_MODEL" \
  --model-label base.en \
  --quality-policy learned \
  --capture-device deterministic_replay \
  --output-dir "$RESULTS_DIR"
```

Explain admitted and rejected events separately. An admitted event must have
`asr_ran=true`; a rejected event must have `asr_ran=false`.

## Optional iPhone Continuity stream

First enumerate devices on the Mac and use the actual audio index shown:

```bash
"$REPO/scripts/stream_mac_to_ubuntu.sh" --list-devices
```

Then set deployment-specific placeholders and stream with rule policy:

```bash
MAC_AUDIO_INDEX="<verified-audio-index>"
UBUNTU_SSH="user@ubuntu-host"
REMOTE_REPO="/path/to/cpp-edge-audio"

"$REPO/scripts/stream_mac_to_ubuntu.sh" \
  --device-index "$MAC_AUDIO_INDEX" \
  --ssh-destination "$UBUNTU_SSH" \
  --remote-repo "$REMOTE_REPO" \
  --remote-demo runtime/cpp/build/robot_hearing_demo \
  --remote-model vendor/whisper.cpp/models/ggml-base.en.bin \
  --quality-policy rule \
  --capture-device iphone_continuity_mic \
  >"$RESULTS_DIR/live.stdout.jsonl" \
  2>"$RESULTS_DIR/live.stderr.log"
```

Wait five full seconds in silence after capture starts. Say `stop`,
`go forward`, `go backward`, `turn left`, and `turn right`, keeping each
compound command continuous and leaving about two seconds between commands.
Do not promise that every live phrase will transcribe canonically.

After the final command, wait three seconds. Press `q`, then Enter if FFmpeg has
interactive terminal control; otherwise press Control-C once. Broken-pipe
diagnostics are benign only when they occur during intentional shutdown.

## Failure-safe behavior

- Rejected audio emits an event but skips ASR.
- An admitted malformed transcript emits `UNKNOWN` with `action:null`.
- No finalized event means no robot action.
- Never describe malformed ASR text remaining `UNKNOWN` as a router failure.

## Presentation fallback order

1. Use deterministic base.en/rule replay.
2. If base cannot load because of resource limits, repeat deterministic replay
   with `TINY_MODEL` and `--model-label tiny.en`.
3. If Wi-Fi or Continuity capture fails, stay on deterministic local replay.
4. If live ASR produces malformed text, show the safe `UNKNOWN` result and
   return to deterministic replay.
5. If CUDA is unavailable, disclose the environment issue rather than quoting
   CPU measurements as GPU latency.

Deterministic replay exits after fixture EOF. For live streaming, use the
shutdown sequence above and retain stdout/stderr separately.

See [technical validation](robot_hearing_validation.md) for evidence and
[interview notes](robot_hearing_interview_notes.md) for the presentation
narrative.
