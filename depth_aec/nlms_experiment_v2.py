"""
AEC via NLMS, experiment 2: does genuine speech survive?

Experiment 1 showed the filter removes what correlates with the reference
and leaves uncorrelated noise alone. The case that actually matters is
harder: when a real person is talking at the same time as the echo, does
the filter preserve their voice or cancel that too?

  near_end_voice = a different sentence entirely, standing in for the user
  mic_signal     = echo + near_end_voice + noise

Both signals are loaded from disk rather than synthesized. Piper is not
deterministic across calls, so caching them keeps every run comparable and
means a change in the numbers reflects a change in the code.

Run with:  uv run depth_aec/nlms_experiment_v2.py
"""

import numpy as np
import sounddevice as sd

from nlms_experiment import erle_db, make_fake_echo, nlms_cancel

NOISE_LEVEL = 0.01
NEAR_END_LEVEL = 1.0
FILTER_LENGTH = 200
SAMPLE_RATE = 22050

reference = np.load("depth_aec/reference.npy")
near_end_voice = np.load("depth_aec/near_end_voice.npy")


def main() -> None:
    # Two different sentences, so two different lengths. Truncating both to
    # the shorter one is enough for this.
    min_len = min(len(reference), len(near_end_voice))
    ref = reference[:min_len]
    voice_clip = near_end_voice[:min_len]

    echo = make_fake_echo(ref)
    rng = np.random.default_rng(0)
    mic_signal = echo + voice_clip * NEAR_END_LEVEL + NOISE_LEVEL * rng.standard_normal(len(echo))

    residual = nlms_cancel(mic_signal, ref)

    print(f"ERLE: {erle_db(mic_signal, residual):.2f} dB")

    # How closely the residual tracks the genuine voice. Close to 1 means
    # the speech came through intact. The first FILTER_LENGTH samples are
    # skipped because the filter never touches them.
    correlation = np.corrcoef(voice_clip[FILTER_LENGTH:], residual[FILTER_LENGTH:])[0][1]
    print(f"Correlation between near_end_voice and residual: {correlation:.3f}")

    print("Playing near_end_voice alone (ground truth)...")
    sd.play(voice_clip.astype(np.float32), SAMPLE_RATE, blocking=True)

    print("Playing mic_signal (echo + speech + noise)...")
    sd.play(mic_signal.astype(np.float32), SAMPLE_RATE, blocking=True)

    print("Playing residual (should sound like near_end_voice)...")
    sd.play(residual.astype(np.float32), SAMPLE_RATE, blocking=True)


if __name__ == "__main__":
    main()
