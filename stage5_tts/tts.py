"""
Stage 5: streaming TTS.

Piper hands back one audio chunk per sentence, so playback can start on the
first sentence while the rest is still being synthesised. The timestamps
printed here are the point: chunks arrive spread out over time rather than
all at once at the end.

Each chunk carries its own sample rate, which is whatever the voice model
was trained at (22050Hz for this voice) and not the 16kHz the rest of the
pipeline runs at.

Run with:  uv run stage5_tts/tts.py
"""

import time

import sounddevice as sd
from piper import PiperVoice

VOICE_MODEL_PATH = "stage5_tts/voices/en_US-lessac-medium.onnx"

TEXT = (
    "Hello there. This is a test of streaming text to speech. "
    "Each sentence should start playing as soon as it is ready, "
    "not after the entire text has been synthesized."
)


def main() -> None:
    voice = PiperVoice.load(VOICE_MODEL_PATH)

    for chunk in voice.synthesize(TEXT):
        print(f"chunk received at {time.time():.3f} ({chunk.sample_rate}Hz)")

        # sd.play returns immediately, so wait for this sentence to finish
        # before synthesising and playing the next one.
        sd.play(chunk.audio_float_array, chunk.sample_rate)
        sd.wait()


if __name__ == "__main__":
    main()
