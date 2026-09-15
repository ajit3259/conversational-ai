"""
Measure the speaker-to-mic echo delay from a pass D capture.

Pass D saves the raw mic and speaker signals to debug_aec_capture.npz on
exit. Cross-correlating the two gives the delay between writing audio and
the mic hearing it, which is what ECHO_DELAY_MS in assistant_aec_48k.py has
to match. On the machine this was built on it came to 294ms, far beyond the
filter's 100ms reach, and with no compensation the canceller did nothing.

For a clean measurement: say one sentence, stay silent while the assistant
answers, then Ctrl+C. Talking over the reply muddies the correlation.

Run with:  uv run stage6_orchestration/measure_echo_delay.py
"""

import numpy as np
from scipy.signal import correlate, correlation_lags, resample_poly

CAPTURE = "stage6_orchestration/debug_aec_capture.npz"
MODEL_RATE = 16000
MAX_DELAY_S = 1.0
MARGIN_MS = 20  # start the filter window this far before the direct path


def main() -> None:
    data = np.load(CAPTURE)
    rate = int(data["rate"])
    down = rate // MODEL_RATE
    mic = resample_poly(data["mic"].astype(np.float64), 1, down)
    ref = resample_poly(data["ref"].astype(np.float64), 1, down)

    if np.abs(ref).max() < 1e-4:
        raise SystemExit("The capture has no playback in it. Let the assistant answer first.")

    xc = correlate(mic, ref, mode="full")
    lags = correlation_lags(len(mic), len(ref), mode="full")
    window = (lags >= 0) & (lags < int(MAX_DELAY_S * MODEL_RATE))
    lag = lags[window][np.argmax(np.abs(xc[window]))]

    # How much of the mic is explained by a scaled, shifted copy of the
    # speaker signal. Well above zero means the echo is mostly linear and a
    # linear filter can cancel it once aligned.
    strength = np.abs(xc[window]).max() / (np.linalg.norm(mic) * np.linalg.norm(ref))

    delay_ms = 1000 * lag / MODEL_RATE
    print(f"echo delay:        {delay_ms:.0f} ms ({lag} samples at {MODEL_RATE}Hz)")
    print(f"correlation peak:  {strength:.2f}")
    print(f"suggested setting: ECHO_DELAY_MS = {max(0, round(delay_ms) - MARGIN_MS)}")


if __name__ == "__main__":
    main()
