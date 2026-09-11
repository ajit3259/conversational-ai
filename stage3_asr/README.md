# Stage 3 · Speech recognition

Buffers audio while somebody is speaking and transcribes the whole utterance
once they stop.

## Run

```bash
uv run stage3_asr/asr.py
```

First run downloads faster-whisper `small`, which takes a while. Stop with
Ctrl+C.

## What to expect

```
speech started
speech stopped
 What time does the train leave?
```

Each utterance is also written to `debug_last_utterance.wav` in the repo
root, so the exact audio handed to Whisper can be played back. That file is
gitignored. Listening to it is the fastest way to tell a bad transcript
apart from a bad recording — if the wav sounds clipped at the start, the
bug is in the buffering, not the model.

## What this is not

This is segment-then-transcribe, not streaming recognition. Nothing is
transcribed until the turn is over, so the whole transcription time lands in
the gap before a reply. Real streaming recognition emits guesses as words
arrive and revises them; Whisper's architecture is not built for that.

Also worth noticing: proper nouns fail silently. Whisper heard "Ajit" and
wrote "Adith", with no flag and no confidence score attached.

## What it cannot do

Answer. That is [stage 4](../stage4_llm).
