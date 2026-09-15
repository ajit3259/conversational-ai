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
speaker. That file played back clean while live playback crackled, which at
first looked like a hardware limit. Passes D and E found otherwise.

## Pass D — native rate, and what was actually wrong

```bash
uv run stage6_orchestration/assistant_aec_48k.py
```

An investigation more than an improvement, kept because of what it found.
Every figure below was measured, not assumed:

- **The crackle was a missed deadline.** The per-sample filter took the full
  32ms of a 32ms block under load. A late callback underruns the speaker, so
  the echo canceller was producing the glitches. Rewriting it as vectorised
  block updates took it to about 0.4ms and the crackle went away.
- **The rewrite diverged at first.** Summing a block of gradients and
  normalising by one window's energy made each step hundreds of times too
  large. Only comparing against the offline reference in `depth_aec/` caught
  it.
- **The filter cancelled nothing because the echo was too late.** On this
  laptop the mic hears the speaker 294ms after it plays; the filter could see
  100ms back. `measure_echo_delay.py` reads the capture pass D saves and
  reports the delay. Compensating for it took ERLE from about 0dB to +7 to
  +10dB.
- **Resampling and non-linearity were ruled out** by replaying the same
  recording through both candidates.

The constants in this file are measured on one machine and should be treated
that way. It still hears itself at +10dB, because a linear filter alone is
never enough — which is where pass E comes in.

## Pass E — WebRTC's echo canceller

```bash
uv run stage6_orchestration/assistant_webrtc_apm.py
```

The same pipeline with echo cancellation handed to WebRTC's audio processing
module, the code Chrome and Google Meet use, via the prebuilt wheel in the
`livekit` package. Nothing to compile.

On the same recording from pass D:

| Canceller | ERLE |
| --- | --- |
| Our filter, delay-compensated | +7.6 dB |
| WebRTC, no delay hint | +37.7 dB |
| WebRTC, measured delay as a hint | +43.9 dB |
| WebRTC, hint and noise suppression | +55.2 dB |

The gap is the layers we never built. WebRTC finds the delay itself — note
+37.7dB with no hint at all — and after its linear filter it suppresses
whatever echo is left, which is what actually makes a call sound clean.

It also takes the lesson from pass D structurally. The callback runs every
10ms and does only what has to happen on the audio clock: play the next
frame, and feed the speaker and mic signals to the canceller at the same
instant. Voice detection moved to its own thread, so a slow call there delays
a decision instead of glitching the audio.
