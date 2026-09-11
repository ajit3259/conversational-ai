# Stage 2 · Voice activity detection

Runs Silero VAD on every block and prints when speech starts and stops.

## Run

```bash
uv run stage2_vad/vad.py
```

First run downloads the Silero model through `torch.hub`. Stop with Ctrl+C.

## What to expect

```
speech started: 2026-07-06 14:22:01.114
speech stopped: 2026-07-06 14:22:04.503
```

Detecting the *start* is easy. Detecting the *stop* is the whole problem,
and it is why `HANGOVER_CHUNKS` exists: the code waits for 12 consecutive
quiet blocks — about 384ms — before deciding the turn is over.

Worth trying: pause mid-sentence for about half a second while reading
something aloud. It will call the turn over while you are still talking.
Nothing in the audio distinguishes "thinking" from "finished", so the
threshold is a guess, and every value is wrong for somebody.

## Note on block size

Silero requires exactly 512 samples per call at 16kHz, which is why
`BLOCK_SIZE` is 512 here rather than the round 30ms of stage 1. The capture
settings are dictated by the models downstream of them.

## What it cannot do

Anything with the speech it detected. That is [stage 3](../stage3_asr).
