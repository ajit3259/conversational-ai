"""
AEC via NLMS, experiment 1: cancelling a known, fabricated echo.

Before pointing an adaptive filter at messy real mic audio, it is worth
checking it works when we control the ground truth. Here the echo path is
one we made up ourselves -- a fixed delay and attenuation -- so if the
filter cannot cancel this, nothing else matters.

  reference  = a synthesized sentence, standing in for what we send to the
               speaker (the one signal we always know exactly)
  echo       = reference pushed through the fabricated room path
  mic_signal = echo + a little noise, standing in for the microphone input

Run with:  uv run depth_aec/nlms_experiment.py
"""

import numpy as np
import sounddevice as sd
from piper import PiperVoice

VOICE_MODEL_PATH = "stage5_tts/voices/en_US-lessac-medium.onnx"

# Taps: how many past reference samples the filter looks at to predict the
# echo now. 200 covers 12.5ms, which is fine for a fabricated delay and
# nowhere near enough for a real room.
FILTER_LENGTH = 200
STEP_SIZE = 0.1  # mu, the adaptation rate
EPSILON = 1e-3  # keeps the normalisation from dividing by zero in quiet passages

FAKE_ECHO_DELAY_SAMPLES = 40
FAKE_ECHO_ATTENUATION = 0.6
NOISE_LEVEL = 0.01


def make_fake_echo(reference: np.ndarray) -> np.ndarray:
    """A known, simple echo path: delay and attenuate. Stands in for the
    real room's unknown and far messier response.
    """
    echo = np.zeros_like(reference)
    echo[FAKE_ECHO_DELAY_SAMPLES:] = (
        reference[: len(reference) - FAKE_ECHO_DELAY_SAMPLES] * FAKE_ECHO_ATTENUATION
    )
    return echo


def nlms_cancel(mic_signal: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Cancel the echo of `reference` out of `mic_signal`, returning the
    residual.
    """
    n_samples = len(mic_signal)
    w = np.zeros(FILTER_LENGTH)  # start knowing nothing about the room
    residual = np.zeros(n_samples)

    for n in range(FILTER_LENGTH, n_samples):
        x_window = reference[n - FILTER_LENGTH : n]
        echo_prediction = np.dot(w, x_window)

        # The error is both this sample's cleaned output and the signal used
        # to correct the filter.
        error = mic_signal[n] - echo_prediction
        residual[n] = error

        # Dividing by the window's own energy is the "normalised" part of
        # NLMS: it keeps the step size sane whether the reference is loud or
        # almost silent.
        w = w + (STEP_SIZE * error * x_window) / (np.dot(x_window, x_window) + EPSILON)

    return residual


def erle_db(mic_signal: np.ndarray, residual: np.ndarray) -> float:
    """Echo Return Loss Enhancement in dB: how much echo energy went away.
    Higher is better.
    """
    return 10 * np.log10(np.mean(mic_signal**2) / np.mean(residual**2))


def main() -> None:
    voice = PiperVoice.load(VOICE_MODEL_PATH)
    chunks = list(voice.synthesize("The quick brown fox jumps over the lazy dog."))
    reference = np.concatenate([c.audio_float_array for c in chunks]).astype(np.float64)
    sample_rate = chunks[0].sample_rate

    echo = make_fake_echo(reference)
    rng = np.random.default_rng(0)
    mic_signal = echo + NOISE_LEVEL * rng.standard_normal(len(echo))

    residual = nlms_cancel(mic_signal, reference)
    print(f"ERLE: {erle_db(mic_signal, residual):.2f} dB")

    print("Playing mic_signal (with echo)...")
    sd.play(mic_signal.astype(np.float32), sample_rate)
    sd.wait()

    print("Playing residual (after AEC)...")
    sd.play(residual.astype(np.float32), sample_rate)
    sd.wait()


if __name__ == "__main__":
    main()
