"""
Stage 6, pass D: the same duplex pipeline, run at the device's native rate.

Pass C opened a duplex stream at 16kHz because that is what VAD and Whisper
want. The hardware here runs at 48kHz, so CoreAudio had to sample-rate
convert both directions underneath us, on a duplex stream, while being fed
512-sample blocks that do not divide evenly into its own buffering. The
persistent crackle in pass C was blamed on duplex audio being hard. This
file tests a more specific theory: that we were asking the device for a rate
it does not run at.

What changes:

  - The stream opens at 48000Hz in 1536-sample blocks. 1536 at 48kHz is the
    same 32ms of audio as 512 at 16kHz, and it divides by exactly 3.
  - The mic block is downsampled 3:1 to 16kHz for AEC, VAD and Whisper.
  - Echo cancellation still runs at 16kHz, on both signals downsampled from
    the same callback invocation, so they stay aligned.
  - Piper's output is resampled to 48kHz once per utterance rather than
    per block, so there are no chunk-boundary artifacts on the output path.
  - Output is scaled by OUTPUT_GAIN. Piper peaks at exactly 1.0, and
    sending full-scale audio to a DAC leaves no room for whatever the
    hardware does next.

Run with:  uv run stage6_orchestration/assistant_aec_48k.py
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
from scipy.signal import resample_poly

# The device's native rate. Check yours with sd.query_devices() -- if it is
# not 48000, change this and DOWNSAMPLE to match.
DEVICE_RATE = 48000
MODEL_RATE = 16000
DOWNSAMPLE = DEVICE_RATE // MODEL_RATE

BLOCK_SIZE_MODEL = 512  # Silero's required block at 16kHz
BLOCK_SIZE_DEVICE = BLOCK_SIZE_MODEL * DOWNSAMPLE  # 1536, still 32ms

# Piper peaks at full scale. Leave the DAC some headroom; set to 1.0 to A/B.
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

FILTER_LENGTH = 1600  # ~100ms of echo tail at 16kHz
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

# Holds samples at DEVICE_RATE now, not MODEL_RATE.
output_buffer: deque = deque()
output_recording: list = []


class IncrementalAEC:
    """NLMS with Geigel double-talk detection, one block at a time.

    Unchanged from pass C. It still runs at 16kHz: cancelling at 48kHz would
    need three times the taps for the same echo tail, and each tap costs
    three times as many multiplies, so roughly nine times the work for no
    extra benefit.
    """

    def __init__(self):
        self.w = np.zeros(FILTER_LENGTH)
        self.ref_history = np.zeros(FILTER_LENGTH)

    def process_block(self, mic_block: np.ndarray, ref_block: np.ndarray) -> np.ndarray:
        combined_ref = np.concatenate([self.ref_history, ref_block])
        n_block = len(mic_block)
        residual = np.zeros(n_block)

        for i in range(n_block):
            x_window = combined_ref[i : i + FILTER_LENGTH]
            echo_prediction = np.dot(self.w, x_window)
            error = mic_block[i] - echo_prediction
            residual[i] = error

            double_talk = abs(mic_block[i]) > GEIGEL_THRESHOLD * max(abs(x_window))
            if not double_talk:
                self.w = self.w + (STEP_SIZE * error * x_window) / (
                    np.dot(x_window, x_window) + EPSILON
                )

        self.ref_history = combined_ref[-FILTER_LENGTH:]
        return residual


aec = IncrementalAEC()


def to_device_rate(audio: np.ndarray, orig_sr: int) -> np.ndarray:
    """Piper's rate to the device's rate, once per utterance."""
    return resample_poly(audio.astype(np.float64), DEVICE_RATE, orig_sr)


def to_model_rate(block: np.ndarray) -> np.ndarray:
    """One 48kHz block down to 16kHz for the models.

    Per-block polyphase resampling has small edge effects at block
    boundaries, since each call filters its block in isolation. It is
    audible to a spectrum plot and not to Whisper.
    """
    return resample_poly(block, 1, DOWNSAMPLE)


def speak(text: str) -> None:
    all_audio = []
    for chunk in tts_voice.synthesize(text):
        all_audio.append(to_device_rate(chunk.audio_float_array, chunk.sample_rate))
    output_buffer.extend((np.concatenate(all_audio) * OUTPUT_GAIN).tolist())

    while output_buffer:
        time.sleep(0.02)


def audio_callback(indata: np.ndarray, outdata: np.ndarray, frames: int, _time, status) -> None:
    if status:
        print(status)

    n_available = min(frames, len(output_buffer))
    popped = [output_buffer.popleft() for _ in range(n_available)]
    ref_block_device = np.array(popped + [0.0] * (frames - n_available))
    outdata[:, 0] = ref_block_device.astype(np.float32)
    output_recording.append(outdata[:, 0].copy())

    mic_block_device = indata[:, 0].astype(np.float64).copy()

    t0 = time.time()
    # Both signals come from this same callback invocation and go through the
    # same filter, so downsampling them together preserves their alignment.
    mic_block = to_model_rate(mic_block_device)
    ref_block = to_model_rate(ref_block_device)
    t1 = time.time()

    if n_available > 0:
        cleaned_block = aec.process_block(mic_block, ref_block)
    else:
        cleaned_block = mic_block
    t2 = time.time()

    audio_tensor = torch.from_numpy(cleaned_block).float()
    speech_prob = vad_model(audio_tensor, MODEL_RATE).item()
    t3 = time.time()

    budget_ms = 1000 * frames / DEVICE_RATE
    spent_ms = 1000 * (t3 - t0)
    if spent_ms > budget_ms:
        print(
            f"OVER BUDGET: resample={1000 * (t1 - t0):.1f}ms "
            f"aec={1000 * (t2 - t1):.1f}ms vad={1000 * (t3 - t2):.1f}ms "
            f"budget={budget_ms:.1f}ms"
        )

    if not state["is_speaking"]:
        if speech_prob > SPEECH_THRESHOLD:
            print("speech started")
            state["is_speaking"] = True
            state["silence_chunks"] = 0
            state["audio_buffer"] = [audio_tensor]

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
            samplerate=DEVICE_RATE,
            blocksize=BLOCK_SIZE_DEVICE,
            channels=1,
            dtype="float32",
            callback=audio_callback,
        ):
            print(f"Listening at {DEVICE_RATE}Hz... talk into the mic.")
            while True:
                audio_chunks = utterance_queue.get()
                transcript = transcribe(audio_chunks)
                print(transcript)
                reply = stream_chat(transcript, conversation_history)
                speak(reply)
    finally:
        if output_recording:
            full_output = np.concatenate(output_recording)
            sf.write(
                "stage6_orchestration/debug_live_output_48k.wav", full_output, DEVICE_RATE
            )
            print("Saved stage6_orchestration/debug_live_output_48k.wav")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
