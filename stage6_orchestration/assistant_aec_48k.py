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

# Consecutive speech blocks required to believe a barge-in while the
# assistant is talking. One block of residual echo is indistinguishable
# from speech; 160ms of it is not. A person interrupting easily clears
# this, a convergence transient does not.
BARGE_IN_CHUNKS = 5

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

# How long after we write audio the mic actually hears it: output buffering,
# the DAC, the air, the ADC and input buffering added together. Measured by
# cross-correlating a recording of this machine's speaker and mic, which came
# to 294ms -- nearly three times the filter's 100ms reach, so the filter was
# looking for the echo in a window it never arrived in and cancelled nothing.
# Delaying the reference lines the two up and leaves the filter only the room
# tail to model. Set 20ms short of the measured delay so the direct path lands
# inside the window. Device- and latency-specific: re-measure with
# measure_echo_delay.py on other hardware.
ECHO_DELAY_MS = 274
ECHO_DELAY_SAMPLES = MODEL_RATE * ECHO_DELAY_MS // 1000

# The weights update once per SUB_BLOCK samples rather than once per sample.
# Measured offline against the per-sample filter in depth_aec/: a whole
# 512-sample block per update loses about 6dB of cancellation, 64 gets it
# back (15.97dB against the reference's 16.37dB) and still costs a twentieth
# of a millisecond. STEP_SIZE is larger than the per-sample version's 0.1
# because the update is now normalised as an average gradient.
SUB_BLOCK = 64
STEP_SIZE = 4.0
EPSILON = 1e-3
GEIGEL_THRESHOLD = 0.75

vad_model, _utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True
)
asr_model = WhisperModel("small", device="cpu", compute_type="int8")
tts_voice = PiperVoice.load(VOICE_MODEL_PATH)

state = {
    "is_speaking": False,
    "silence_chunks": 0,
    "audio_buffer": [],
    "barge_in_candidate": [],
    "echo_tail": 0,
}
utterance_queue = queue.Queue()

# Holds samples at DEVICE_RATE now, not MODEL_RATE.
output_buffer: deque = deque()
output_recording: list = []
erle_log: list = []

# Raw device-rate mic and speaker blocks, saved on exit so the real echo can
# be analysed offline: its delay, whether the per-sample filter can cancel
# it at all, and what per-block resampling does to it.
capture_mic: list = []
capture_ref: list = []


class BlockNLMS:
    """NLMS with Geigel double-talk detection, updating once per block.

    Pass C walked the block sample by sample, and that is what broke the
    real-time budget. The prediction loop was never the problem -- it costs
    about half a millisecond. The weight update was, because it builds a
    fresh 1600-element array 512 times per block, which came to roughly 23ms
    in isolation and the full 32ms under load. A callback that misses its
    deadline makes the output underrun, so the algorithm meant to clean up
    the audio was itself producing glitches.

    Both halves are really convolutions, and numpy will do them in one call:

      prediction  y[i] = dot(w, ref[i : i+FILTER_LENGTH])
                       = correlate(ref, w, "valid")
      gradient    g[j] = sum_i e[i] * ref[i+j]
                       = correlate(ref, e, "valid")

    Same arithmetic to floating-point precision, about 50x faster. The
    tradeoff is real but small: weights now adapt once per 32ms block rather
    than every sample, so the filter converges a little more slowly. This is
    Block LMS, and it is what practical echo cancellers use, usually in the
    frequency domain.

    Still running at 16kHz. Cancelling at 48kHz would need three times the
    taps for the same echo tail, at three times the cost each.
    """

    def __init__(self):
        self.w = np.zeros(FILTER_LENGTH)
        self.ref_history = np.zeros(FILTER_LENGTH)

    def process_block(self, mic_block: np.ndarray, ref_block: np.ndarray) -> np.ndarray:
        residual = np.empty(len(mic_block))

        for k in range(0, len(mic_block), SUB_BLOCK):
            mic = mic_block[k : k + SUB_BLOCK]
            ref = ref_block[k : k + SUB_BLOCK]
            n = len(mic)

            combined_ref = np.concatenate([self.ref_history, ref])
            echo_prediction = np.correlate(combined_ref, self.w, mode="valid")[:n]
            error = mic - echo_prediction
            residual[k : k + n] = error

            # Geigel, judged once per sub-block: is the mic louder than this
            # stretch of reference could explain as echo?
            # Nothing to learn from a silent reference, and it now runs on every
            # block rather than only during playback.
            reference_silent = np.abs(combined_ref).max() < 1e-4
            double_talk = np.abs(mic).max() > GEIGEL_THRESHOLD * np.abs(combined_ref).max()
            if not reference_silent and not double_talk:
                gradient = np.correlate(combined_ref, error, mode="valid")[:FILTER_LENGTH]
                # Normalise by the energy one window sees, times the number of
                # samples summed into the gradient -- that makes this the
                # average per-sample update rather than the sum of them. Miss
                # this and the step is hundreds of times too large, and the
                # filter diverges instead of converging.
                window_energy = np.dot(combined_ref, combined_ref) * FILTER_LENGTH / len(combined_ref)
                self.w += STEP_SIZE * gradient / (n * window_energy + EPSILON)

            self.ref_history = combined_ref[-FILTER_LENGTH:]

        return residual


class DelayLine:
    """Delays a stream by a fixed number of samples, block in, block out."""

    def __init__(self, delay: int):
        self.pending = np.zeros(delay)

    def process(self, block: np.ndarray) -> np.ndarray:
        if len(self.pending) == 0:
            return block
        joined = np.concatenate([self.pending, block])
        out, self.pending = joined[: len(block)], joined[len(block) :]
        return out


aec = BlockNLMS()
ref_delay = DelayLine(ECHO_DELAY_SAMPLES)


def to_device_rate(audio: np.ndarray, orig_sr: int) -> np.ndarray:
    """Piper's rate to the device's rate, once per utterance."""
    return resample_poly(audio.astype(np.float64), DEVICE_RATE, orig_sr)


def to_model_rate(block: np.ndarray) -> np.ndarray:
    """One device-rate block down to 16kHz for the models.

    Per-block polyphase resampling has small edge effects at block
    boundaries, since each call filters its block in isolation. It is
    audible to a spectrum plot and not to Whisper.

    Setting DEVICE_RATE to 16000 makes this a no-op, which turns this file
    into a 16kHz control for the same fast filter.
    """
    if DOWNSAMPLE == 1:
        return block
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
    capture_mic.append(indata[:, 0].copy())
    capture_ref.append(ref_block_device.astype(np.float32))

    t0 = time.time()
    # Both signals come from this same callback invocation and go through the
    # same filter, so downsampling them together preserves their alignment.
    mic_block = to_model_rate(mic_block_device)
    ref_block = to_model_rate(ref_block_device)
    t1 = time.time()

    # What we play now reaches the mic ECHO_DELAY later, so the filter is fed
    # the reference from that long ago rather than the block just written.
    delayed_ref = ref_delay.process(ref_block)

    # The echo keeps arriving for ECHO_DELAY plus the room tail after the
    # speaker goes quiet. Until then the assistant is still audible, even
    # though the output buffer is already empty.
    if n_available > 0:
        state["echo_tail"] = ECHO_DELAY_SAMPLES + FILTER_LENGTH
    else:
        state["echo_tail"] = max(0, state["echo_tail"] - len(mic_block))
    assistant_audible = n_available > 0 or state["echo_tail"] > 0

    # Cheap enough to run on every block now, which matters: skipping it
    # whenever nothing is playing would leave the delayed echo uncancelled.
    cleaned_block = aec.process_block(mic_block, delayed_ref)

    if np.abs(delayed_ref).max() > 1e-4:
        # How much echo went away, in dB. Positive means the filter is
        # cancelling; around zero means it is doing nothing.
        mic_energy = np.mean(mic_block**2)
        residual_energy = np.mean(cleaned_block**2)
        if mic_energy > 1e-9:
            erle_log.append(10 * np.log10(mic_energy / (residual_energy + 1e-12)))
        if len(erle_log) >= 4:
            print(f"  ERLE {np.mean(erle_log):+.1f} dB   mic {np.sqrt(mic_energy):.4f}")
            erle_log.clear()
    else:
        erle_log.clear()
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
            if assistant_audible:
                # The assistant is talking. The filter starts each utterance
                # knowing nothing about the current echo path, so for the
                # first stretch of playback the echo comes through barely
                # cancelled -- and one block of leaked echo looks exactly
                # like speech. Require a run of detections before believing
                # a barge-in, keeping the blocks so the interruption is not
                # clipped when we do believe it.
                state["barge_in_candidate"].append(audio_tensor)
                if len(state["barge_in_candidate"]) < BARGE_IN_CHUNKS:
                    return
                print("barge-in")
                output_buffer.clear()
                state["audio_buffer"] = state["barge_in_candidate"]
                state["barge_in_candidate"] = []
            else:
                print("speech started")
                state["audio_buffer"] = [audio_tensor]

            state["is_speaking"] = True
            state["silence_chunks"] = 0
        else:
            # Any quiet block breaks the run.
            state["barge_in_candidate"] = []
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
        if capture_mic:
            np.savez(
                "stage6_orchestration/debug_aec_capture.npz",
                mic=np.concatenate(capture_mic),
                ref=np.concatenate(capture_ref),
                rate=DEVICE_RATE,
            )
            print("Saved stage6_orchestration/debug_aec_capture.npz")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
