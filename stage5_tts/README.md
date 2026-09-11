# Stage 5 · Speech synthesis

Synthesizes speech with Piper, playing each sentence as soon as it is ready.

## Prerequisites

Voice models are not committed — they are about 60MB each. Fetch the one
this stage uses:

```bash
uv run python -m piper.download_voices en_US-lessac-medium --download-dir stage5_tts/voices
```

Other voices work the same way, for example:

```bash
uv run python -m piper.download_voices hi_IN-pratham-medium --download-dir stage5_tts/voices
```

Then point `VOICE_MODEL_PATH` at whichever one you downloaded.

## Run

```bash
uv run stage5_tts/tts.py
```

## What to expect

Three sentences spoken, with a timestamp printed as each chunk arrives:

```
chunk received at 1751790123.412 (22050Hz)
chunk received at 1751790125.008 (22050Hz)
```

The timestamps are the point. They are spread out over time, which means
playback starts on sentence one while sentence two is still being
synthesized.

## The sample rate trap

Note the 22050Hz in that output. The voice model renders at the rate it was
trained at, while the rest of this pipeline runs at 16kHz. Feed one into the
other without converting and speech comes out too fast and too high — the
assistant sounds like a chipmunk. Nothing errors, so the only symptom is a
silly voice. [Stage 6's AEC build](../stage6_orchestration) has to resample
for exactly this reason.

## What it cannot do

Run as part of a live conversation. That is
[stage 6](../stage6_orchestration).
