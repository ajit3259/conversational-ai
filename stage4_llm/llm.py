"""
Stage 4: streaming LLM dialogue turn.

Sends a transcript to a local model over the OpenAI-compatible streaming
endpoint and prints tokens as they arrive rather than all at once at the
end. Conversation history is passed back in full on every call, since the
model keeps nothing between requests.

Run with:  uv run stage4_llm/llm.py
"""

import json

import requests

LLM_URL = "http://localhost:11434/v1/chat/completions"
MODEL = "gemma3n:e4b"

# The prompt has to constrain the answer for the ear rather than the eye:
# markdown and emoji become noise once a TTS model reads them out, sometimes
# character by character. The requirement comes from the *next* stage.
SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You are a helpful voice assistant. Answer in 1-3 short spoken "
        "sentences. Do not use markdown, bullet points, asterisks, or "
        "emoji, since your response will be read aloud by a "
        "text-to-speech system."
    ),
}


def stream_chat(user_message: str, conversation_history: list) -> str:
    conversation_history.append({"role": "user", "content": user_message})
    payload = {
        "model": MODEL,
        "messages": [SYSTEM_PROMPT, *conversation_history],
        "stream": True,
        # Gemma leaks its reasoning into the content field otherwise.
        "reasoning_effort": "none",
    }

    # stream=True stops requests from downloading the whole body up front,
    # which is what makes reading it incrementally possible.
    response = requests.post(LLM_URL, json=payload, stream=True)
    response.encoding = "utf-8"
    response.raise_for_status()

    full_text = ""
    for line in response.iter_lines(decode_unicode=True):
        # Server-sent events arrive as "data: {...}", separated by blank
        # lines, and end with a "data: [DONE]" that is not JSON.
        line = line[5:].strip() if line else ""
        if line in ("", "[DONE]"):
            continue

        chunk = json.loads(line)
        token = chunk["choices"][0]["delta"].get("content", "")
        print(token, end="", flush=True)
        full_text += token

    print()
    conversation_history.append({"role": "assistant", "content": full_text})
    return full_text


if __name__ == "__main__":
    history = []
    stream_chat("My favourite color is red.", history)
    stream_chat("What is my favourite color?", history)
