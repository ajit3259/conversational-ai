# Stage 7 · The same thing, assembled

Everything stages 1-6 built by hand, using the same four models, wired
together by [Pipecat](https://github.com/pipecat-ai/pipecat) instead of by
us. Roughly 100 lines against about 300, and the seams are somebody else's
problem.

This is the honest ending to the exercise. You would not hand-roll any of
the earlier stages for real work. Doing it once is how you find out what the
framework is actually doing for you.

## Prerequisites

Ollama, as before:

```bash
ollama pull gemma3n:e4b
ollama serve
```

The Piper voice is fetched automatically into `../stage5_tts/voices` if it
is not already there.

## Run

This is a **separate uv project** from the repo root, on purpose — the
hand-rolled stages pin their own torch and faster-whisper, and there is no
reason to force those two dependency sets to agree.

```bash
cd stage7_pipecat
uv run bot.py
```

Then open <http://localhost:7860> and allow microphone access. First run
downloads Whisper and Silero weights, so give it a minute.

## What the framework is doing for you

Each of these was a section of the write-up and a real problem in the
hand-rolled build.

| Seam we hit | What happens here |
| --- | --- |
| The assistant hears itself | The browser captures audio, so `getUserMedia` gives echo cancellation for free |
| When is the turn over? | Handled, with the VAD attached to the conversation rather than the microphone |
| Both sides stream, the joint does not | The LLM streams into TTS sentence by sentence |
| Whose sample rate wins? | Resampling between services is internal |
| Mic block size and rate | Decided by the transport |

The echo one is the most interesting. Stage 6 pass C spent its time on an
adaptive filter and then hit a wall that no filter could fix, because the
hardware's duplex distortion is non-linear. That problem does not appear
here at all — not because Pipecat solved it, but because the browser owns
the microphone now, and browsers have solved it for years.

## One detail worth noticing in the code

The VAD is attached to `LLMUserAggregatorParams`, not to the transport:

```python
user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
    context,
    user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
)
```

In our build, the VAD lived in the audio callback, right next to the
microphone, because it felt like an audio concern. Pipecat puts it next to
the thing that decides when a user's turn is complete.

That placement is the argument the write-up makes, expressed as an import
path. *Is there sound* and *is it my turn* are different questions, and the
framework files the second one under conversation rather than audio.

## Friction, since the write-up promised honesty about it

On Apple Silicon, Pipecat's Whisper module imports `mlx_whisper` at module
level whether or not you use the MLX backend, so the module will not import
without it. The dependency is declared with a platform marker in
`pyproject.toml` so this resolves itself, but it is a reminder that reaching
for the framework does not mean reaching for zero problems — only for
different ones.
