"""
AEC via NLMS, experiment 3: double-talk detection.

Experiment 2 exposed the double-talk problem: when a loud independent voice
arrives alongside the echo, the filter tries to explain that voice as echo
and corrupts its own weights. The fix is to notice when it is happening and
stop learning, while carrying on cancelling with the weights we already
have.

The Geigel detector is the classic cheap version: if the mic sample is
louder than the recent reference could plausibly explain as echo, somebody
else must be talking.

Run with:  uv run depth_aec/nlms_experiment_v3_doubletalk.py
"""

import numpy as np
import sounddevice as sd

from nlms_experiment import (
    EPSILON,
    FILTER_LENGTH,
    STEP_SIZE,
    erle_db,
    make_fake_echo,
)
from nlms_experiment_v2 import (
    NEAR_END_LEVEL,
    NOISE_LEVEL,
    SAMPLE_RATE,
    near_end_voice,
    reference,
)

# Expected maximum echo gain. The fabricated path attenuates by 0.6, so
# anything above this fraction of the loudest recent reference sample is
# louder than echo alone could account for.
GEIGEL_THRESHOLD = 0.75


def nlms_cancel_with_dtd(mic_signal: np.ndarray, reference: np.ndarray) -> np.ndarray:
    n_samples = len(mic_signal)
    w = np.zeros(FILTER_LENGTH)
    residual = np.zeros(n_samples)
    frozen_count = 0

    for n in range(FILTER_LENGTH, n_samples):
        x_window = reference[n - FILTER_LENGTH : n]
        echo_prediction = np.dot(w, x_window)
        error = mic_signal[n] - echo_prediction
        residual[n] = error

        double_talk = abs(mic_signal[n]) > GEIGEL_THRESHOLD * max(abs(x_window))

        # Cancelling continues either way; only the learning stops.
        if double_talk:
            frozen_count += 1
        else:
            w = w + (STEP_SIZE * error * x_window) / (np.dot(x_window, x_window) + EPSILON)

    adapted = n_samples - FILTER_LENGTH
    print(f"Adaptation frozen on {frozen_count}/{adapted} samples "
          f"({100 * frozen_count / adapted:.1f}%)")
    return residual


def main() -> None:
    min_len = min(len(reference), len(near_end_voice))
    ref = reference[:min_len]
    voice_clip = near_end_voice[:min_len]

    echo = make_fake_echo(ref)
    rng = np.random.default_rng(0)
    mic_signal = echo + voice_clip * NEAR_END_LEVEL + NOISE_LEVEL * rng.standard_normal(len(echo))

    residual = nlms_cancel_with_dtd(mic_signal, ref)

    print(f"ERLE: {erle_db(mic_signal, residual):.2f} dB")
    correlation = np.corrcoef(voice_clip[FILTER_LENGTH:], residual[FILTER_LENGTH:])[0][1]
    print(f"Correlation between near_end_voice and residual: {correlation:.3f}")

    print("Playing near_end_voice alone (ground truth)...")
    sd.play(voice_clip.astype(np.float32), SAMPLE_RATE, blocking=True)

    print("Playing residual (after AEC + double-talk detection)...")
    sd.play(residual.astype(np.float32), SAMPLE_RATE, blocking=True)


if __name__ == "__main__":
    main()
