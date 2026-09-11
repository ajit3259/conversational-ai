# Stage 6 · Orchestration

Three passes over the same problem, kept as separate files so each one can
be run and compared rather than hidden behind a flag.

## Prerequisites

Everything the earlier stages needed:

```bash
ollama pull gemma3n:e4b && ollama serve
uv run python -m piper.download_voices en_US-lessac-medium --download-dir stage5_tts/voices
```

## Pass A — sequential turn-taking

```bash
uv run stage6_orchestration/assistant.py
```

Stages 2-5 wired into one loop. Speak, wait, get an answer, repeat.

The new idea is the hand-off. The mic callback runs on its own thread and
must return within one 32ms block, so it cannot transcribe or call an LLM.
It puts finished utterances on a `queue.Queue` and the main thread does the
slow work.

It also mutes the microphone while speaking. Without that, the assistant
hears itself through the speakers and answers its own voice — which is the
entire subject of passes B and C. The cost is that you cannot interrupt it.

## Pass B — barge-in

```bash
uv run stage6_orchestration/assistant_bargein.py
```

Keeps the microphone live during playback and treats detected speech as an
interruption: stop playback, start a new utterance from the triggering
block.

**Wear headphones for this one.** On speakers it will interrupt itself
constantly, which is expected rather than a bug — the interruption logic is
correct, and what it is reacting to is its own voice.

One ordering detail matters. `sd.stop()` is what wakes the main thread out
of `sd.wait()`, so `assistant_speaking` has to be cleared *before* that
call, or the waking thread reads a value that is about to change.

## Pass C — duplex with echo cancellation

```bash
uv run stage6_orchestration/assistant_aec.py
```

Cancels the assistant's own voice out of the mic signal instead of muting
around it, using the filter developed in [depth_aec](../depth_aec).

Two changes fall out of that requirement rather than being chosen:

- Input and output become one duplex `sd.Stream`. Cancellation needs the
  reference and the mic audio sample-aligned, and two separate streams have
  two separate clocks.
- One stream means one sample rate. The mic side is pinned to 16kHz by VAD
  and Whisper, so Piper's 22050Hz output has to be resampled down.

The payoff is that barge-in stops being a special case. With the echo
removed, interrupting is just "speech detected, so clear the output buffer".

Watch for `OVER BUDGET` lines. Those mean the per-block work took longer
than the 32ms a block covers, which is the pipeline falling behind the
microphone.

On exit it writes `debug_live_output.wav` — the exact samples sent to the
speaker. If that file plays back clean but live playback crackled, the
problem is the hardware's duplex handling rather than anything in this code.
That is what happened here, and a linear filter structurally cannot cancel
non-linear hardware distortion.
