"""
Stage 5: Streaming TTS

Goal: synthesize speech from text using Piper, playing each sentence's
audio as soon as it's ready rather than waiting for the whole text to be
synthesized -- the TTS-side mirror of Stage 4's token streaming.

Fill in the TODOs. Run with:  uv run stage5_tts/tts.py
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
    # Plumbing: loading the voice model is done for you.
    voice = PiperVoice.load(VOICE_MODEL_PATH)

    # TODO 1: call voice.synthesize(TEXT) -- it returns a generator of
    # AudioChunk objects, one per sentence. Iterate over it.
    for chunk in voice.synthesize(TEXT):
        # TODO 2: print a timestamp (e.g. time.time()) right when you
        # receive each chunk, so we can later verify sentences are arriving
        # (and being played) incrementally rather than all at once.
        print("Received audio chunk at: ", time.time())

        # TODO 3: play the chunk's audio using sounddevice. Hint:
        # sd.play(chunk.audio_float_array, chunk.sample_rate) starts
        # playback, but returns immediately (non-blocking) -- you need
        # sd.wait() right after to block until that chunk finishes playing
        # before moving on to synthesize/play the next one.
        sd.play(chunk.audio_float_array, chunk.sample_rate)
        sd.wait()


if __name__ == "__main__":
    main()
