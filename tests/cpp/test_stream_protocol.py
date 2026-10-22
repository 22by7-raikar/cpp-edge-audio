#!/usr/bin/env python3
"""End-to-end check of the voice_command_stream stdout protocol.

argv[1]: stream_protocol_driver (real VoiceCommandLoop, fake transcriber)
argv[2]: voice_command_stream    (checked for argument validation only)
"""

import json
import subprocess
import sys


def check_event_stream(driver: str) -> None:
    completed = subprocess.run([driver], check=False, text=True, capture_output=True)
    assert completed.returncode == 0, f"driver failed: {completed.stderr}"
    assert completed.stderr == "", f"unexpected stderr: {completed.stderr!r}"

    events = [json.loads(line) for line in completed.stdout.splitlines()]
    assert len(events) == 2, f"expected two events, got {len(events)}"
    first, second = events

    assert first["event"] == "voice_command"
    assert first["utterance_id"] == 1
    assert first["finalization_reason"] == "end_silence"
    assert first["intent"] == "LOOK_LEFT"
    assert first["action"] == "look_left"

    assert second["utterance_id"] == 2
    assert second["finalization_reason"] == "end_of_stream"
    assert second["intent"] == "UNKNOWN"
    assert second["action"] is None

    for event in events:
        assert event["capture_device"] == "protocol_fixture"
        assert event["transport"] == "test_callback"
        assert event["quality_policy"] == "rule"
        assert event["quality_threshold"] == 0.3
        assert event["admitted"] is True
        assert event["asr_ran"] is True
        assert event["stream"]["frames_processed"] > 0
        assert event["latency"]["quality_us"] >= 0.0
        assert event["latency"]["post_utterance_ms"] >= 0.0


def check_argument_validation(cli: str) -> None:
    # Invalid arguments must fail before the model is loaded and print nothing on stdout.
    invalid = subprocess.run(
        [cli, "--rate", "16000", "--channels", "3", "--model", "must-not-be-loaded.bin"],
        check=False, text=True, capture_output=True,
    )
    assert invalid.returncode == 2, invalid.returncode
    assert invalid.stdout == ""
    assert "--channels must be 1 or 2" in invalid.stderr
    assert "[asr]" not in invalid.stderr


def main() -> int:
    check_event_stream(sys.argv[1])
    check_argument_validation(sys.argv[2])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
