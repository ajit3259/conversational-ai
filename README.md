# conversational-ai

A cascaded voice assistant built from scratch, one stage at a time, to see
what is actually involved between the models rather than inside them.

Nothing here is production code. Every model used is off the shelf and
swappable; the point of the repo is the wiring between them.

## Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Two things are downloaded on first use rather than committed:

**Piper voice models** (~60MB each), needed by stages 5 and 6:

```bash
uv run python -m piper.download_voices en_US-lessac-medium --download-dir stage5_tts/voices
```

**A local LLM via [Ollama](https://ollama.com)**, needed by stages 4 and 6:

```bash
ollama pull gemma3n:e4b
ollama serve        # listens on localhost:11434
```

Silero VAD and faster-whisper pull their own weights automatically the first
time a stage that needs them runs, so expect the first run to be slow.

## The stages

Each is standalone and meant to be read in order. Every stage exists because
the previous one ran into a problem it could not solve.

| Stage | What it adds | Needs |
| --- | --- | --- |
| [1 · mic capture](stage1_mic_capture) | Audio as a stream of numbers | mic |
| [2 · VAD](stage2_vad) | Knowing when somebody is speaking | mic |
| [3 · ASR](stage3_asr) | Turning an utterance into text | mic |
| [4 · LLM](stage4_llm) | A streamed reply | Ollama |
| [5 · TTS](stage5_tts) | Turning text back into sound | voice model |
| [6 · orchestration](stage6_orchestration) | All of it, live, with interruption | all of the above |
| [depth: AEC](depth_aec) | Cancelling the assistant's own voice | — |

`DOUBTS.md` is a running log of questions raised while working through each
stage, with the answers where there are any.
