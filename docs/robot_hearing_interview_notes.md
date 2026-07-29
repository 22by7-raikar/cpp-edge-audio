# Robot Hearing interview notes

## Thirty-second explanation

Robot Hearing is a Linux-first streaming command pipeline around whisper.cpp.
It converts raw PCM into completed utterances, evaluates audio quality with the
existing DSP and ordered 27-feature model, conditionally runs Whisper exactly
once, and maps only five canonical transcripts to robot actions. Every
completed utterance produces one JSON event; rejected audio skips ASR and
malformed text remains `UNKNOWN`. Deterministic base.en replay on CUDA is
accepted for presentation, while live iPhone streaming remains experimental.

## Two-minute walkthrough

1. A bounded decoder converts mono or stereo S16LE stdin into normalized mono
   samples without accumulating the stream.
2. Streaming VAD and the utterance state machine add pre-roll, require a speech
   start trigger, and finalize on endpoint silence or a duration limit.
3. Only a completed utterance enters the same DSP and ordered 27-feature
   analysis used by the file path.
4. Rule, learned, or hybrid policy makes one admission decision. The learned
   threshold remains frozen at `0.3`.
5. Rejection emits JSON immediately with `asr_ran=false`.
6. Admission calls Whisper once, then a conservative exact-match router
   normalizes safe textual variation and maps only five command phrases.
7. JSONL stays on stdout; diagnostics and backend evidence stay on stderr.
8. base.en was selected for the GPU presentation because it routed 12/15 pilot
   commands versus 7/15 for tiny.en, with a modest measured latency increase.

## Five-minute deep-dive outline

1. State the contract: one completed utterance, one admission decision, zero or
   one Whisper call, one JSON event.
2. Draw the capture, decoder, VAD, segmentation, quality, ASR, routing, and
   event boundaries.
3. Explain why frame VAD answers “is speech active now?” while quality
   admission answers “is this completed utterance worth transcribing?”
4. Show the admitted and rejected branches and their ASR-call counts.
5. Explain ordered feature parity and why learned inference cannot run on
   individual frames.
6. Present the tiny/base pilot table and its limited scope.
7. Discuss endpoint latency versus split/merge risk using the measured
   600 ms endpoint and 740 ms internal pause.
8. Show deterministic replay evidence, CUDA0/no-fallback evidence, and
   JSONL/stderr separation.
9. Describe the exact-byte live/replay experiment and five-second warm-up.
10. Close with current limitations and the next validation work.

## Why quality-gate audio before Whisper?

Whisper is robust, but robustness does not make every captured segment worth
processing. A quality gate can:

- avoid unnecessary compute for clearly unusable or non-speech segments;
- reduce downstream work before an expensive ASR call;
- expose an explicit, measurable admission operating point; and
- make rejection behavior observable in logs and tests.

The limitation matters as much as the benefit: a poorly calibrated gate can
reject valid short commands. That is why rule policy is the live presentation
choice and why the learned gate is not claimed to be calibrated for this
deployment domain.

## VAD versus quality admission

VAD is a streaming temporal decision. It identifies speech-active frames and
determines when an utterance starts and ends. Quality admission is a
completed-utterance decision over file-equivalent DSP and aggregate features.
VAD cannot replace the quality model, and the quality model cannot safely
replace frame-level endpointing.

## Why analysis waits for finalization

The learned model was trained on one ordered 27-feature vector per complete
file. Frame-by-frame inference would change feature meaning, DSP context, and
the model’s deployment distribution. Waiting for finalization preserves
training/runtime semantics and lets rule, learned, and hybrid policies share
one analyzer.

## Why ASR runs exactly once

Admission and transcription are separate operations. The completed utterance
is analyzed once, then a single guarded wrapper either skips Whisper or invokes
it once on the admitted audio. Tests count callback invocations for both
branches. This prevents frame-level or policy-comparison paths from
accidentally transcribing the same utterance multiple times.

## Why routing is conservative

The router normalizes case, surrounding and repeated whitespace, and ordinary
terminal punctuation, then performs an exact lookup. It intentionally avoids
substring matching, edit distance, fuzzy aliases, and semantic inference.
`Go back, Lord.` must remain `UNKNOWN`; turning malformed ASR into motion would
be less safe than declining to act.

## Model choice

base.en routed 12/15 expected command positions in the accepted pilot, compared
with 7/15 for tiny.en. Median GPU ASR increased from 16.0 ms to 23.8 ms, which
was an acceptable presentation tradeoff. tiny.en remains the lower-resource
fallback.

small.en was not selected as the presentation baseline because the accepted
deployment comparison and repeatable runbook were finalized around base.en.
The evidence here does not claim that base.en is universally more accurate than
small.en; it chooses the smallest validated model that materially improved the
pilot routing result.

## CPU versus CUDA latency

The accepted GPU pilot measured ASR in tens of milliseconds, with model-load
warm-up visible in the upper tail. Separate CPU smoke tests were in the
hundreds of milliseconds. Those runs used different fixtures and scopes, so
they demonstrate deployment tradeoffs but are not an apples-to-apples hardware
benchmark. GPU latency must be measured in the ordinary Linux environment
where CUDA is actually visible.

## Endpoint tradeoff

The current endpoint is 600 ms. A real 740 ms pause inside “turn right” caused
a correct split. Raising the endpoint to 760–800 ms would preserve that example
but add 160–200 ms of post-speech waiting and increase the chance of merging
nearby commands. Three recordings are not enough data to select a new value,
so the configuration stayed unchanged.

## Learned-gate domain shift

The frozen model is numerically stable, but calibration is a separate question.
Short live commands, devices, rooms, and speaking styles differ from the
training distribution. Stable probabilities on identical audio prove
determinism, not calibration. A larger labeled live-command corpus is required
before changing the threshold or recommending learned admission for live use.

## What failed and what the evidence showed

- The five-second JFK smoke scored about `0.0626` under learned policy. Direct
  completed-buffer parity showed the rejection was legitimate, not an
  integration mismatch.
- Command routing initially failed on case, whitespace, and terminal
  punctuation. Conservative normalization and regression tests fixed it
  without adding fuzzy behavior.
- CUDA failed inside the restricted development sandbox because the GPU was not
  exposed. The same binary used CUDA0 with no CPU fallback in the ordinary
  Linux terminal.
- One recording split “turn right.” A production VAD trace found a real 740 ms
  internal pause, so the 600 ms endpoint behaved correctly.
- Early live attempts missed the first event. Exact PCM capture, identical-byte
  replay, and two five-second warm-up runs found no transport or segmenter
  defect and produced five events.
- Five events did not imply five correct actions: live ASR remained variable,
  while malformed transcripts safely stayed `UNKNOWN`.

## Honest future work

- Collect a larger labeled corpus of live commands across speakers, devices,
  distances, rooms, and noise conditions.
- Recalibrate or retrain quality admission only with an explicit new protocol;
  do not tune on the current pilot.
- Add a capture-ready handshake that distinguishes model, transport, and audio
  readiness.
- Evaluate command ASR separately from segmentation and routing.
- Instrument full capture-to-action latency with synchronized timestamps.
- Add a ROS2 action adapter only with allowlists, watchdogs, timeouts, and
  robot-side safety interlocks.

## Likely interview questions

**Why not send every frame to Whisper?**

Whisper expects meaningful audio windows, repeated calls waste compute, and
frame-level calls destroy the one-utterance/one-event contract.

**Why not use fuzzy routing to recover ASR errors?**

The output controls a robot. A false negative is observable and safe; a guessed
motion command can be unsafe.

**Does 12/15 prove base.en is accurate?**

No. It is a pilot selection result on three identical recordings, not a
statistical accuracy estimate.

**Why keep the learned gate if rule policy is the demo default?**

It validates the completed-utterance architecture and provides a measurable
admission mechanism, but its live calibration still needs data.

**Did SSH drop the first command?**

The controlled `tee` capture and identical-byte replay produced matching event
boundaries and outputs. The five-second startup protocol produced five events
twice, so no transport-loss defect was found.

**What does `post_utterance_ms` measure?**

Processing after finalization: quality analysis, admitted ASR, routing, and
event construction. It is not microphone-to-action latency.

**What would you ship next?**

First add readiness and end-to-end observability, then collect labeled
live-command data. Only after that would I change calibration, endpointing, or
robot integration.

See [technical validation](robot_hearing_validation.md) and the
[presentation runbook](robot_hearing_demo.md).
