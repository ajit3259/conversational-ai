# Stage 4 · LLM dialogue turn

Sends a transcript to a local model and prints tokens as they arrive.

## Prerequisites

```bash
ollama pull gemma3n:e4b
ollama serve        # localhost:11434
```

Any OpenAI-compatible endpoint works — change `LLM_URL` and `MODEL`. This
stage is the most swappable in the pipeline; changing the backend touches
this file and nothing else.

## Run

```bash
uv run stage4_llm/llm.py
```

Runs two hardcoded turns and exits.

## What to expect

Text appearing word by word rather than all at once, then a second turn
proving history carried over:

```
Red is a nice choice. I'll remember that.
Your favourite color is red.
```

The model keeps nothing between calls, so the whole prior exchange is sent
back every turn. That list is the only thing making this a conversation.

## The prompt is doing real work

The system prompt forbids markdown, bullets and emoji, and holds the answer
to a couple of short sentences. None of that is for the LLM's benefit — it
is for the speech synthesiser in [stage 5](../stage5_tts), which will read
asterisks out character by character and make a screen-length paragraph
unbearable to sit through.

The requirement comes from the stage *after* this one. In a cascade the
audio flows forwards and the constraints travel backwards.
