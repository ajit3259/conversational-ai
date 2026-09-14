"""
Stage 1: mic capture.

Opens the default input device and prints the loudness of each incoming
block. The point is to watch audio arrive as a steady drip of fixed-size
number arrays: the hardware clock decides when the callback fires, not
anything about the sound itself. Silence is not an absence of blocks, it
is blocks full of near-zero samples.

Run with:  uv run stage1_mic_capture/capture.py
"""

import math

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000  # 16kHz: the rate the speech models downstream expect
BLOCK_DURATION_MS = 30
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_DURATION_MS / 1000)

# log10 runs off to -inf as rms approaches zero, and "silence" on a real mic
# is just float noise near zero. Clamp before the log so the dB reading stays
# meaningful instead of swinging wildly.
RMS_FLOOR = 1e-8


def audio_callback(indata: np.ndarray, _frames: int, _time, status) -> None:
    if status:
        print(status)

    # indata is (frames, channels), float32 in [-1, 1]. A single sample says
    # nothing about loudness -- it is one instantaneous amplitude, and it can
    # be negative. RMS collapses the whole block into one number: square to
    # drop the sign, average, then root back to the original scale.
    rms = np.sqrt(np.mean(indata**2))

    # dBFS is measured against the loudest representable signal (amplitude
    # 1.0), so real audio always reads negative. 0 dBFS is the clipping
    # ceiling, not a target.
    dbfs = 20 * math.log10(max(rms, RMS_FLOOR))
    print(f"rms {rms:.4f}   dBFS {dbfs:6.1f}")


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
