#!/usr/bin/env python3

import json
import subprocess
import sys


def main() -> int:
    completed = subprocess.run(
        [sys.argv[1]],
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"protocol driver failed: {completed.returncode}\n"
            f"stderr={completed.stderr}"
        )
    if completed.stderr:
        raise AssertionError(f"unexpected stderr: {completed.stderr!r}")

    lines = completed.stdout.splitlines()
    if len(lines) != 2:
        raise AssertionError(f"expected two JSON events, got {len(lines)}")
    events = [json.loads(line) for line in lines]

    first, second = events
    assert first["event"] == "robot_voice_command"
    assert first["utterance_id"] == 1
    assert first["finalization_reason"] == "end_silence"
    assert first["asr_ran"] is True
    assert first["intent"] == "LOOK_LEFT"
    assert first["action"] == "look_left"

    assert second["utterance_id"] == 2
    assert second["finalization_reason"] == "end_of_stream"
    assert second["asr_ran"] is True
    assert second["intent"] == "UNKNOWN"
    assert second["action"] is None

    for event in events:
        assert event["capture_device"] == "protocol_fixture"
        assert event["transport"] == "test_callback"
        assert event["quality_policy"] == "rule"
        assert event["quality_threshold"] == 0.3
        assert event["admitted"] is True
        assert event["stream"]["invalid_frames"] == 0
        assert event["latency"]["quality_us"] >= 0.0
        assert event["latency"]["post_utterance_ms"] >= 0.0

    invalid = subprocess.run(
        [
            sys.argv[2],
            "--source",
            "stdin",
            "--input-rate",
            "16000",
            "--input-channels",
            "1",
            "--quality-policy",
            "learned",
            "--quality-threshold",
            "0.4",
            "--model",
            "must-not-be-loaded.bin",
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    assert invalid.returncode != 0
    assert invalid.stdout == ""
    assert "requires --quality-threshold 0.3" in invalid.stderr
    assert "loading model" not in invalid.stderr
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
