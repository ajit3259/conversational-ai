"""
Stage 6, pass A: sequential turn-taking.

Wires stages 2-5 into one loop: mic -> VAD -> buffer -> transcribe -> LLM
-> TTS, then back to listening. No interruption yet.

The new idea here is the hand-off. The mic callback runs on its own thread
and has to return within one block's worth of time, so it cannot transcribe
or call an LLM. It drops a finished utterance on a queue instead and the
main thread does the slow work.

Playback also has to mute the microphone, or the assistant hears itself
through the speakers and answers its own voice. That is a real problem, not
a shortcut, and passes B and C deal with it properly.

Run with:  uv run stage6_orchestration/assistant.py
"""

import json
import queue

import numpy as np
import requests
import sounddevice as sd
import torch
from faster_whisper import WhisperModel
from piper import PiperVoice

SAMPLE_RATE = 16000
BLOCK_SIZE = 512
SPEECH_THRESHOLD = 0.5
HANGOVER_CHUNKS = 12

LLM_URL = "http://localhost:11434/v1/chat/completions"
MODEL = "gemma3n:e4b"
SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You are a helpful voice assistant. Answer in 1-3 short spoken "
        "sentences. Do not use markdown, bullet points, asterisks, or "
        "emoji, since your response will be read aloud by a "
        "text-to-speech system."
    ),
}

VOICE_MODEL_PATH = "stage5_tts/voices/en_US-lessac-medium.onnx"

vad_model, _utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True
)
asr_model = WhisperModel("small", device="cpu", compute_type="int8")
tts_voice = PiperVoice.load(VOICE_MODEL_PATH)

state = {
    "is_speaking": False,
    "silence_chunks": 0,
    "audio_buffer": [],
    "assistant_speaking": False,
}

utterance_queue = queue.Queue()


def audio_callback(indata: np.ndarray, _frames: int, _time, status) -> None:
    # Deaf while talking. Crude, and the reason passes B and C exist.
    if state["assistant_speaking"]:
        return

    if status:
        print(status)

    audio_tensor = torch.from_numpy(indata.copy()).squeeze()
    speech_prob = vad_model(audio_tensor, SAMPLE_RATE).item()

    if not state["is_speaking"]:
        if speech_prob > SPEECH_THRESHOLD:
            print("speech started")
            state["is_speaking"] = True
            state["silence_chunks"] = 0
            state["audio_buffer"] = [audio_tensor]
    else:
        state["audio_buffer"].append(audio_tensor)

        if speech_prob > SPEECH_THRESHOLD:
            state["silence_chunks"] = 0
        else:
            state["silence_chunks"] += 1
            if state["silence_chunks"] >= HANGOVER_CHUNKS:
                print("speech stopped")
                state["is_speaking"] = False
                # Hand off rather than transcribe here: this thread has to
                # be ready for the next block in 32ms.
                utterance_queue.put(state["audio_buffer"])
                state["audio_buffer"] = []


def transcribe(audio_chunks: list) -> str:
    full_audio = np.concatenate(audio_chunks)
    segments, _ = asr_model.transcribe(full_audio, language="en")
    return " ".join(segment.text for segment in segments).strip()


def stream_chat(user_message: str, conversation_history: list) -> str:
    conversation_history.append({"role": "user", "content": user_message})
    payload = {
        "model": MODEL,
        "messages": [SYSTEM_PROMPT, *conversation_history],
        "stream": True,
    }
    response = requests.post(LLM_URL, json=payload, stream=True)
    response.encoding = "utf-8"
    response.raise_for_status()

    full_text = ""
    for line in response.iter_lines(decode_unicode=True):
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


def speak(text: str) -> None:
    state["assistant_speaking"] = True
    for chunk in tts_voice.synthesize(text):
        sd.play(chunk.audio_float_array, chunk.sample_rate)
        sd.wait()
    state["assistant_speaking"] = False


def main() -> None:
    conversation_history = []

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        channels=1,
        dtype="float32",
        callback=audio_callback,
    ):
        print("Listening... talk into the mic.")
        while True:
            audio_chunks = utterance_queue.get()
            transcript = transcribe(audio_chunks)
            print(transcript)
            reply = stream_chat(transcript, conversation_history)
            speak(reply)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
