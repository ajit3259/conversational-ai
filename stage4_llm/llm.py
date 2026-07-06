"""
Stage 4: LLM dialogue turn (streaming)

Goal: send a transcript to the local Gemma model over the OpenAI-compatible
streaming API, and print each token as it arrives (not all at once at the
end), so you can see/feel the responsiveness streaming buys you.

Fill in the TODOs. Run with:  uv run stage4_llm/llm.py
"""

import json
import requests

LLM_URL = "http://localhost:11434/v1/chat/completions"
MODEL = "gemma3n:e4b"
SYSTEM_PROMPT = {"role": "system", "content": "You are a helpful voice assistant. Answer in 1-3 short spoken sentences. Do not use markdown, bullet points, asterisks, or emoji, since your response will be read aloud by a text-to-speech system."}

def stream_chat(user_message: str, conversation_history: list) -> str:
    conversation_history.append({"role": "user", "content": user_message})
    payload = {
        "model": MODEL,
        "messages": [
            SYSTEM_PROMPT,
            *conversation_history,
        ],
        "stream": True,
        "reasoning_effort": "none"
    }

    # Plumbing: opening the streaming HTTP request is done for you.
    # `stream=True` tells `requests` not to download the whole body up front --
    # you read it incrementally instead, which is what makes streaming possible.
    response = requests.post(LLM_URL, json=payload, stream=True)
    response.encoding = "utf-8"
    response.raise_for_status()

    full_text = ""

    # TODO 1: iterate over the response body line by line. Hint:
    # `response.iter_lines(decode_unicode=True)` gives you one line (str) at
    # a time. Each Server-Sent-Event line looks like: `data: {...json...}`,
    # with blank lines in between events (skip those), and a final line
    # `data: [DONE]` marking the end of the stream (stop on that, don't
    # try to json.loads it).
    for line in response.iter_lines(decode_unicode=True):
        # TODO 2: strip the "data: " prefix, json.loads the remainder to get
        # the chunk dict, then pull out the text piece via:
        #   chunk["choices"][0]["delta"].get("content", "")
        # (delta may not always have "content", e.g. the very first chunk
        # only carries {"role": "assistant"} with no content yet)
        # print(line)
        line = line[5:].strip() # removing front data:
        if line == "[DONE]" or line == "": # do not json load last [DONE]
            continue
        json_response = json.loads(line)
        # print(json_response)
        token = json_response["choices"][0]["delta"].get("content", "")

        # TODO 3: print the token immediately (no newline, flush=True so it
        # shows up right away rather than being buffered), and also
        # append it to `full_text` so we have the complete reply once the
        # loop ends -- Stage 5 (TTS) will need this complete text.
        print(token, end="", flush=True)
        full_text += token

    print()  # newline after the streamed reply finishes
    conversation_history.append({"role": "assistant", "content": full_text})
    return full_text


if __name__ == "__main__":
    conversation_history = []
    stream_chat("My favourite color is red.", conversation_history)
    print(conversation_history)
    stream_chat("What is my favourite color?", conversation_history)
    print(conversation_history)

