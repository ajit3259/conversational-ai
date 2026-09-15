"""
Stage 6, pass E: full duplex with WebRTC's echo canceller.

Passes C and D built echo cancellation by hand and got as far as the linear
filter. This pass hands the job to WebRTC's audio processing module -- the
same code that cancels echo in Chrome and Google Meet -- through the prebuilt
wheel LiveKit ships, so there is nothing to compile.

What WebRTC does that our filter did not:

  - Finds the echo delay on its own and keeps tracking it. Pass D had to
    measure 294ms by cross-correlating a recording and hardcode it.
  - Suppresses whatever echo the linear filter leaves behind. This is the
    layer that makes the difference: on the same recording from pass D our
    filter reached +7.6dB, WebRTC reaches +43.9dB, and +55.2dB with noise
    suppression on.
  - Noise suppression, a high-pass filter and gain control, as switches.

And one structural change, from the lesson pass D taught: the audio callback
now does only what has to happen on the audio clock -- playback, and feeding
both sides of the conversation to the canceller at the same instant. Voice
detection moved to its own thread. The callback's deadline is 10ms now, and
nothing slow is allowed near it.

Run with:  uv run stage6_orchestration/assistant_webrtc_apm.py
"""

import json
import queue
import threading
import time
from collections import deque

import numpy as np
import requests
import sounddevice as sd
import torch
from faster_whisper import WhisperModel
from livekit import rtc
from piper import PiperVoice
from scipy.signal import firwin, lfilter, resample_poly

# The device runs at 48kHz, so the stream does too and the canceller works on
# the raw device signal. Only the cleaned mic audio gets downsampled, for the
# models.
DEVICE_RATE = 48000
MODEL_RATE = 16000
DOWNSAMPLE = DEVICE_RATE // MODEL_RATE

# WebRTC processes audio in exactly 10ms frames, so the stream delivers
# exactly one frame per callback.
FRAME_SIZE = DEVICE_RATE // 100  # 480 samples

# Silero still wants 512 samples at 16kHz, which no longer lines up with a
# frame. The listening thread collects cleaned audio until it has one.
VAD_BLOCK = 512

# A hint, not a requirement: WebRTC estimates the delay itself. On pass D's
# recording it reached +37.7dB with no hint and +43.9dB with the measured
# 294ms. Leaving it at 0 means nothing here is tuned to one machine; set it
# from measure_echo_delay.py if you want the head start.
ECHO_DELAY_HINT_MS = 0

# Headroom for the DAC. Piper peaks at full scale.
OUTPUT_GAIN = 0.8

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

apm = rtc.AudioProcessingModule(
    echo_cancellation=True,
    noise_suppression=True,
    high_pass_filter=True,
    auto_gain_control=True,
)

state = {"is_speaking": False, "silence_chunks": 0, "audio_buffer": []}

# Three threads now, each with one job, joined by queues:
#   audio callback  -> clean_mic_queue  -> listening thread
#   listening thread -> utterance_queue -> main thread (transcribe, LLM, speak)
clean_mic_queue: queue.Queue = queue.Queue()
utterance_queue: queue.Queue = queue.Queue()

# Samples waiting to be played, at DEVICE_RATE. Filled by speak(), drained by
# the callback, and cleared by the listening thread on a barge-in.
output_buffer: deque = deque()

# Echo readout, so this pass can be compared directly against pass D.
erle_log: list = []


class Decimator:
    """Downsample a stream by an integer factor, block by block.

    Pass D resampled each block in isolation, which is wrong for a stream:
    every block's filter starts cold. Here the low-pass filter's state is
    carried from one block to the next, so the output is the same as if the
    whole stream had been filtered in one go.
    """

    def __init__(self, factor: int, taps: int = 63):
        self.factor = factor
        # Cut off just below the new Nyquist frequency, so nothing above it
        # folds back down as aliasing.
        self.coeffs = firwin(taps, 0.9 / factor)
        self.filter_state = np.zeros(taps - 1)

    def process(self, block: np.ndarray) -> np.ndarray:
        filtered, self.filter_state = lfilter(self.coeffs, 1.0, block, zi=self.filter_state)
        # Every block is a whole number of frames, and FRAME_SIZE divides by
        # the factor, so keeping every third sample stays in phase across
        # blocks.
        return filtered[:: self.factor]


decimator = Decimator(DOWNSAMPLE)


def to_frame(samples: np.ndarray) -> rtc.AudioFrame:
    """float32 in [-1, 1] -> the int16 frame WebRTC expects."""
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    return rtc.AudioFrame(pcm.tobytes(), DEVICE_RATE, 1, FRAME_SIZE)


def from_frame(frame: rtc.AudioFrame) -> np.ndarray:
    return np.frombuffer(frame.data, dtype=np.int16).astype(np.float32) / 32767


def audio_callback(indata: np.ndarray, outdata: np.ndarray, frames: int, _time, status) -> None:
    if status:
        print(status)
    started = time.perf_counter()

    # 1. What the speaker plays this frame.
    playing = np.zeros(frames, dtype=np.float32)
    for i in range(frames):
        try:
            playing[i] = output_buffer.popleft()
        except IndexError:
            # Empty, or cleared by a barge-in on the other thread.
            break
    outdata[:, 0] = playing

    # 2. Show the canceller the speaker signal and the mic signal together.
    # WebRTC wants the far end first, then the near end that contains its
    # echo -- both from the same instant, which is why this stays in the
    # callback while everything else moves out.
    apm.process_reverse_stream(to_frame(playing))

    mic_frame = to_frame(indata[:, 0])
    apm.set_stream_delay_ms(ECHO_DELAY_HINT_MS)
    apm.process_stream(mic_frame)
    cleaned = from_frame(mic_frame)

    # 3. Hand the cleaned audio off, downsampled for the models.
    clean_mic_queue.put(decimator.process(cleaned))

    if np.abs(playing).max() > 1e-4:
        mic_energy = np.mean(indata[:, 0] ** 2)
        if mic_energy > 1e-9:
            erle_log.append(10 * np.log10(mic_energy / (np.mean(cleaned**2) + 1e-12)))
        if len(erle_log) >= 50:  # every half second
            print(f"  ERLE {np.mean(erle_log):+.1f} dB")
            erle_log.clear()

    spent_ms = 1000 * (time.perf_counter() - started)
    budget_ms = 1000 * frames / DEVICE_RATE
    if spent_ms > budget_ms:
        print(f"OVER BUDGET: {spent_ms:.1f}ms of {budget_ms:.0f}ms")


def listen() -> None:
    """Voice detection and turn-taking, off the audio thread.

    In passes C and D this ran inside the callback. Silero is fast, but its
    first call takes tens of milliseconds, and any stall there is a missed
    audio deadline. Here a slow call just means the queue grows briefly.
    """
    pending = np.zeros(0, dtype=np.float32)
    while True:
        pending = np.concatenate([pending, clean_mic_queue.get()])
        while len(pending) >= VAD_BLOCK:
            block, pending = pending[:VAD_BLOCK], pending[VAD_BLOCK:]
            handle_block(torch.from_numpy(block.astype(np.float32)))


def handle_block(audio_tensor: torch.Tensor) -> None:
    speech_prob = vad_model(audio_tensor, MODEL_RATE).item()

    if not state["is_speaking"]:
        if speech_prob > SPEECH_THRESHOLD:
            # With the echo actually gone, anything VAD hears while the
            # assistant is talking is the user, so barge-in needs no special
            # handling: stop playing and start listening.
            if output_buffer:
                print("barge-in")
                output_buffer.clear()
            else:
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
                utterance_queue.put(state["audio_buffer"])
                state["audio_buffer"] = []


def speak(text: str) -> None:
    all_audio = []
    for chunk in tts_voice.synthesize(text):
        all_audio.append(
            resample_poly(chunk.audio_float_array.astype(np.float64), DEVICE_RATE, chunk.sample_rate)
        )
    output_buffer.extend((np.concatenate(all_audio) * OUTPUT_GAIN).tolist())

    while output_buffer:
        time.sleep(0.02)


def transcribe(audio_chunks: list) -> str:
    full_audio = torch.cat(audio_chunks).numpy()
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
    threading.Thread(target=listen, daemon=True).start()

    with sd.Stream(
        samplerate=DEVICE_RATE,
        blocksize=FRAME_SIZE,
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
