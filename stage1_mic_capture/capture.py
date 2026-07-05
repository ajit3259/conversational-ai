"""
Stage 1: Mic capture

Goal: open the default microphone and print the volume (RMS) of each
incoming audio chunk in real time, so you can see live audio flowing
through your program before we add VAD/ASR/etc on top.

Fill in the TODOs. Run with:  uv run stage1_mic_capture/capture.py
Stop with Ctrl+C.
"""

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000  # samples per second (Hz). 16kHz is standard for speech models.
BLOCK_DURATION_MS = 30  # how many milliseconds of audio per callback
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_DURATION_MS / 1000)


def audio_callback(indata: np.ndarray, frames: int, time, status) -> None:
    if status:
        print(status)

    # TODO 1: compute the RMS (root-mean-square) volume of `indata`.
    # indata has shape (frames, channels) and dtype float32, values in [-1, 1].
    # RMS = sqrt(mean(samples^2))
    rms = ...  # replace with your computation

    # TODO 2: print something like a simple volume meter, e.g.
    # print(f"{rms:.4f}")
    ...


def main() -> None:
    # TODO 3: open a sd.InputStream using SAMPLE_RATE, BLOCK_SIZE, mono (channels=1),
    # dtype="float32", and audio_callback as the callback.
    # Hint: sd.InputStream is a context manager -- use it in a `with` block,
    # then just sleep/wait forever (e.g. `while True: sd.sleep(1000)`) since
    # the callback runs on its own thread.
    ...


if __name__ == "__main__":
    main()
