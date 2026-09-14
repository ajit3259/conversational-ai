"""
Stage 6, pass C: full duplex with acoustic echo cancellation.

Pass B kept the microphone live during playback and paid for it: without
headphones the assistant hears itself and treats its own voice as a
barge-in. This version cancels the echo instead of muting around it, using
the NLMS filter and Geigel double-talk detector from depth_aec/.

Two architectural consequences, both forced by the AEC rather than chosen:

  - Input and output become a single duplex sd.Stream. Subtracting a
    reference from a recording only works if the two are sample-aligned,
    and two independent streams have two independent clocks.
  - Because one stream has one sample rate, and the mic side is pinned to
    16kHz by VAD and Whisper, Piper's 22050Hz output has to be resampled
    down before it can be played.

The payoff is that barge-in stops being a special case. With the echo
actually removed, VAD runs continuously on a clean signal and interrupting
is just "speech detected while audio is queued, so drop the queue".

Run with:  uv run stage6_orchestration/assistant_aec.py
"""

import json
import queue
import time
from collections import deque

import numpy as np
import requests
import sounddevice as sd
import soundfile as sf
import torch
from faster_whisper import WhisperModel
from piper import PiperVoice
from scipy.signal import resample

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

# A real room's echo tail runs far longer than the 200 taps (12.5ms) that
# were enough for the fabricated echo in depth_aec/. 1600 taps covers 100ms.
# An isolated benchmark suggested 3200 was affordable at ~2.6ms per block;
# under real conditions, with torch and VAD competing for CPU, it measured
# ~41ms against a 32ms budget. Real echo coverage against real time.
FILTER_LENGTH = 1600
STEP_SIZE = 0.1
EPSILON = 1e-3
GEIGEL_THRESHOLD = 0.75

vad_model, _utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True
)
asr_model = WhisperModel("small", device="cpu", compute_type="int8")
tts_voice = PiperVoice.load(VOICE_MODEL_PATH)

state = {"is_speaking": False, "silence_chunks": 0, "audio_buffer": []}
utterance_queue = queue.Queue()

# speak() appends samples, the duplex callback pops them off the front every
# block. A deque rather than a list because popping from the front of a list
# is O(n) and this happens 31 times a second.
output_buffer: deque = deque()

# Every block actually written to the speaker, so it can be saved and played
# back later. Comparing that recording against a clean offline render is how
# you tell an algorithm bug apart from a hardware one.
output_recording: list = []


class IncrementalAEC:
    """NLMS with Geigel double-talk detection, one block at a time.

    The offline version in depth_aec/ processes a whole array at once. Here
    the filter weights and a tail of reference history have to survive
    between calls, because a sample near the start of a block still needs
    FILTER_LENGTH samples of history that belong to the previous one.
    """

    def __init__(self):
        self.w = np.zeros(FILTER_LENGTH)
        self.ref_history = np.zeros(FILTER_LENGTH)

    def process_block(self, mic_block: np.ndarray, ref_block: np.ndarray) -> np.ndarray:
        combined_ref = np.concatenate([self.ref_history, ref_block])
        n_block = len(mic_block)
        residual = np.zeros(n_block)

        for i in range(n_block):
            # Index i in the block is FILTER_LENGTH + i in combined_ref,
            # since the first FILTER_LENGTH entries are carried-over history.
            x_window = combined_ref[i : i + FILTER_LENGTH]
            echo_prediction = np.dot(self.w, x_window)
            error = mic_block[i] - echo_prediction
            residual[i] = error

            # Louder than the reference could explain as echo means somebody
            # else is talking, so stop learning but keep cancelling.
            double_talk = abs(mic_block[i]) > GEIGEL_THRESHOLD * max(abs(x_window))
            if not double_talk:
                self.w = self.w + (STEP_SIZE * error * x_window) / (
                    np.dot(x_window, x_window) + EPSILON
                )

        self.ref_history = combined_ref[-FILTER_LENGTH:]
        return residual


aec = IncrementalAEC()


def resample_to_16k(audio: np.ndarray, orig_sr: int) -> np.ndarray:
    """Piper renders at its own rate; the duplex stream runs at 16kHz."""
    n_target = int(len(audio) * SAMPLE_RATE / orig_sr)
    return resample(audio, n_target).astype(np.float64)


def speak(text: str) -> None:
    """Synthesize and resample the whole reply before queueing any of it.

    Doing it up front rules out the producer falling behind the real-time
    callback draining the buffer. It also means nothing is spoken until the
    entire reply exists, which is latency this build never reclaimed.
    """
    all_audio = []
    for chunk in tts_voice.synthesize(text):
        all_audio.append(resample_to_16k(chunk.audio_float_array, chunk.sample_rate))
    output_buffer.extend(np.concatenate(all_audio).tolist())

    while output_buffer:
        time.sleep(0.02)


def audio_callback(indata: np.ndarray, outdata: np.ndarray, frames: int, _time, status) -> None:
    if status:
        print(status)

    # Fill this block's output from the queue, padding with silence.
    n_available = min(frames, len(output_buffer))
    popped = [output_buffer.popleft() for _ in range(n_available)]
    ref_block = np.array(popped + [0.0] * (frames - n_available))
    outdata[:, 0] = ref_block.astype(np.float32)
    output_recording.append(outdata[:, 0].copy())

    mic_block = indata[:, 0].astype(np.float64).copy()

    t0 = time.time()
    if n_available > 0:
        cleaned_block = aec.process_block(mic_block, ref_block)
    else:
        # Nothing playing means no echo to cancel, so skip the expensive
        # per-sample loop entirely.
        cleaned_block = mic_block
    t1 = time.time()

    audio_tensor = torch.from_numpy(cleaned_block).float()
    speech_prob = vad_model(audio_tensor, SAMPLE_RATE).item()
    t2 = time.time()

    budget_ms = 1000 * frames / SAMPLE_RATE
    aec_ms = 1000 * (t1 - t0)
    vad_ms = 1000 * (t2 - t1)
    if aec_ms + vad_ms > budget_ms:
        print(f"OVER BUDGET: aec={aec_ms:.1f}ms vad={vad_ms:.1f}ms budget={budget_ms:.1f}ms")

    if not state["is_speaking"]:
        if speech_prob > SPEECH_THRESHOLD:
            print("speech started")
            state["is_speaking"] = True
            state["silence_chunks"] = 0
            state["audio_buffer"] = [audio_tensor]

            # Barge-in, in one line. With the echo gone this is genuinely
            # just "somebody is talking, so stop playing".
            if len(output_buffer):
                output_buffer.clear()
    else:
        state["audio_buffer"].append(audio_tensor)

        if speech_prob > SPEECH_THRESHOLD:
            state["silence_chunks"] = 0
        else:
            state["silence_chunks"] += 1
            if state["silence_chunks"] >= HANGOVER_CHUNKS:
                print("speech stopped")
                state["is_speaking"] = False
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


def main() -> None:
    conversation_history = []

    try:
        with sd.Stream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            channels=1,
            dtype="float32",
            latency="high",
            callback=audio_callback,
        ):
            print("Listening... talk into the mic.")
            while True:
                audio_chunks = utterance_queue.get()
                transcript = transcribe(audio_chunks)
                print(transcript)
                reply = stream_chat(transcript, conversation_history)
                speak(reply)
    finally:
        if output_recording:
            full_output = np.concatenate(output_recording)
            sf.write("stage6_orchestration/debug_live_output.wav", full_output, SAMPLE_RATE)
            print("Saved stage6_orchestration/debug_live_output.wav")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
