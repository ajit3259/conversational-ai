"""
Stage 3: segment-then-transcribe ASR.

Reuses Stage 2's start/stop detection, but buffers the audio while somebody
is speaking and transcribes the whole utterance once they stop. This is not
true streaming recognition -- nothing is transcribed until the turn is over,
so the wait lands directly in the gap before the reply.

Note the buffer starts with the block that *triggered* detection. Dropping
it clips the first syllable off every utterance, which looks like a bad
speech model rather than a bookkeeping mistake.

Run with:  uv run stage3_asr/asr.py
"""

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000
BLOCK_SIZE = 512  # Silero requirement at 16kHz

SPEECH_THRESHOLD = 0.5
HANGOVER_CHUNKS = 12  # ~384ms of continuous silence to confirm "stopped"

# Written out on every utterance so the exact audio handed to Whisper can be
# played back. Listening to the input is the fastest way to tell a bad
# transcript apart from a bad recording.
DEBUG_WAV = "debug_last_utterance.wav"

vad_model, _utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True
)
asr_model = WhisperModel("small", device="cpu", compute_type="int8")

state = {"is_speaking": False, "silence_chunks": 0, "audio_buffer": []}


def transcribe(audio_chunks: list[np.ndarray]) -> None:
    full_audio = np.concatenate(audio_chunks)
    sf.write(DEBUG_WAV, full_audio, SAMPLE_RATE)

    segments, _ = asr_model.transcribe(full_audio, language="en")
    for segment in segments:
        print(segment.text)


def audio_callback(indata: np.ndarray, _frames: int, _time, status) -> None:
    if status:
        print(status)

    # sounddevice reuses its input buffer between calls, so copy before
    # holding onto it.
    audio_tensor = torch.from_numpy(indata.copy()).squeeze()
    speech_prob = vad_model(audio_tensor, SAMPLE_RATE).item()

    if not state["is_speaking"]:
        if speech_prob > SPEECH_THRESHOLD:
            print("speech started")
            state["is_speaking"] = True
            state["silence_chunks"] = 0
            state["audio_buffer"] = [audio_tensor]
    else:
        # Keep buffering through the quiet tail too -- we only know the turn
        # ended after the fact, and the tail is cheap to include.
        state["audio_buffer"].append(audio_tensor)

        if speech_prob > SPEECH_THRESHOLD:
            state["silence_chunks"] = 0
        else:
            state["silence_chunks"] += 1
            if state["silence_chunks"] >= HANGOVER_CHUNKS:
                print("speech stopped")
                state["is_speaking"] = False
                transcribe(state["audio_buffer"])
                state["audio_buffer"] = []


def main() -> None:
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        channels=1,
        dtype="float32",
        callback=audio_callback,
    ):
        while True:
            sd.sleep(1000)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
