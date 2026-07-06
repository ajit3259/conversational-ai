"""
Stage 3: Streaming ASR (segment-then-transcribe)

Goal: reuse Stage 2's VAD start/stop detection, but this time buffer the
audio while the person is speaking, and once they stop, transcribe the
whole utterance with faster-whisper and print the text.

Fill in the TODOs. Run with:  uv run stage3_asr/asr.py
Stop with Ctrl+C.
"""

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000
BLOCK_SIZE = 512  # Silero VAD requirement at 16kHz

SPEECH_THRESHOLD = 0.5
HANGOVER_CHUNKS = 12  # ~384ms of continuous silence to confirm "stopped"

# Model loading is plumbing, done for you.
vad_model, _utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True
)
asr_model = WhisperModel("small", device="cpu", compute_type="int8")

# TODO 1: extend the state dict from Stage 2 with something to hold the
# buffered audio chunks while the person is speaking (e.g. "audio_buffer": []).
state = {"is_speaking": False, "silence_chunks": 0, "audio_buffer": []}


def transcribe(audio_chunks: list[np.ndarray]) -> None:
    # TODO 2: concatenate the list of chunk arrays into a single 1D float32
    # numpy array. Hint: np.concatenate(...) on the list, then check/adjust
    # shape -- transcribe() wants 1D, but your chunks are (BLOCK_SIZE, 1).
    full_audio = np.concat(audio_chunks)
    print(full_audio.shape)
    sf.write("debug_last_utterance.wav", full_audio, SAMPLE_RATE)

    # TODO 3: call asr_model.transcribe(full_audio) -- it returns
    # (segments, info). Iterate over `segments` and print each segment's
    # `.text`.
    segments, _ = asr_model.transcribe(full_audio, language="en")
    for segment in segments:
        print(segment.text)


def audio_callback(indata: np.ndarray, frames: int, time, status) -> None:
    if status:
        print(status)

    indata_copy = indata.copy()
    audio_tensor = torch.from_numpy(indata_copy).squeeze()
    speech_prob = vad_model(audio_tensor, SAMPLE_RATE).item()

    if not state["is_speaking"]:
        if speech_prob > SPEECH_THRESHOLD:
            print("speech started")
            state["is_speaking"] = True
            state["silence_chunks"] = 0
            # TODO 4: reset/start the audio buffer for this new utterance.
            state["audio_buffer"] = []
            state["audio_buffer"].append(audio_tensor)
    else:
        # TODO 5: regardless of speech or silence, while we're in the
        # "is_speaking" state we still want to keep buffering audio (the
        # silent tail matters less, but simplest is to just always append
        # `indata` to the buffer here before the threshold check below).
        state["audio_buffer"].append(audio_tensor)

        if speech_prob > SPEECH_THRESHOLD:
            state["silence_chunks"] = 0
        else:
            state["silence_chunks"] += 1
            if state["silence_chunks"] >= HANGOVER_CHUNKS:
                print("speech stopped")
                state["is_speaking"] = False
                # TODO 6: call transcribe() with the buffered audio now
                # that the utterance is complete.
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
    main()
