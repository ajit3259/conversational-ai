"""
Stage 2: voice activity detection.

Runs Silero VAD on every incoming block and prints when speech starts and
stops. The model is the easy part. The interesting bit is deciding that
speech has *stopped*, which means waiting out a run of quiet blocks, because
a pause in the middle of a sentence and a pause at the end of one look
exactly the same to the detector.

Run with:  uv run stage2_vad/vad.py
"""

from datetime import datetime

import numpy as np
import sounddevice as sd
import torch

SAMPLE_RATE = 16000
# Silero requires exactly 512 samples per call at 16kHz, which is 32ms.
BLOCK_SIZE = 512

SPEECH_THRESHOLD = 0.5  # probability above which a block counts as speech

# Consecutive quiet blocks before calling the turn over: 12 * 32ms = ~384ms.
# Too low and it interrupts mid-thought, too high and it feels sluggish.
HANGOVER_CHUNKS = 12

model, _utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True
)

# The callback runs on its own thread and cannot rebind names from out here,
# so anything that has to survive between blocks lives in this dict.
state = {"is_speaking": False, "silence_chunks": 0}


def audio_callback(indata: np.ndarray, _frames: int, _time, status) -> None:
    if status:
        print(status)

    # (BLOCK_SIZE, 1) from sounddevice -> (BLOCK_SIZE,) for the model.
    audio_tensor = torch.from_numpy(indata).squeeze()
    speech_prob = model(audio_tensor, SAMPLE_RATE).item()

    # Only the transitions are worth printing; otherwise this fires 31 times
    # a second forever.
    if not state["is_speaking"]:
        if speech_prob > SPEECH_THRESHOLD:
            print(f"speech started: {datetime.now()}")
            state["is_speaking"] = True
            state["silence_chunks"] = 0
    else:
        if speech_prob > SPEECH_THRESHOLD:
            state["silence_chunks"] = 0
        else:
            state["silence_chunks"] += 1
            if state["silence_chunks"] >= HANGOVER_CHUNKS:
                print(f"speech stopped: {datetime.now()}")
                state["is_speaking"] = False


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
