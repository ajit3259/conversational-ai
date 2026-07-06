"""
Stage 2: Voice Activity Detection (VAD)

Goal: instead of just printing volume, use Silero VAD to detect actual
speech, and print "speech started" / "speech stopped" events as you talk.

Fill in the TODOs. Run with:  uv run stage2_vad/vad.py
Stop with Ctrl+C.
"""

import numpy as np
import sounddevice as sd
import torch
from datetime import datetime

SAMPLE_RATE = 16000
# Silero VAD requires exactly 512 samples per call at 16kHz (32ms of audio).
BLOCK_SIZE = 512

SPEECH_THRESHOLD = 0.5  # probability above which we consider a chunk "speech"

HANGOVER_CHUNKS = 12 # consecutive 32ms audio duration for detecting speech pause

# Model loading is plumbing, done for you.
model, _utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True
)

# TODO 1: you'll need some state that persists *between* callback calls, so
# you know whether you were already "in speech" on the previous chunk (to
# detect the *transition* from not-speech -> speech, and speech -> not-speech,
# rather than printing every single chunk). A mutable container like a dict
# or a class works well here since `audio_callback` can't easily reassign a
# plain outer variable. Define that state before `audio_callback`.
state = {"is_speaking": False, "silence_chunks": 0}


def audio_callback(indata: np.ndarray, frames: int, time, status) -> None:
    if status:
        print(status)

    # TODO 2: convert `indata` (numpy array, shape (BLOCK_SIZE, 1)) into the
    # 1D torch.Tensor shape the model expects: (BLOCK_SIZE,).
    # Hint: torch.from_numpy(...) then .squeeze() or reshape.
    audio_tensor = torch.from_numpy(indata).squeeze()
    # print(audio_tensor.shape)

    # TODO 3: call the model to get a speech probability for this chunk.
    # Hint: speech_prob = model(audio_tensor, SAMPLE_RATE).item()
    speech_prob = model(audio_tensor, SAMPLE_RATE).item()
    # print(speech_prob)

    # TODO 4: using `state["is_speaking"]` and `speech_prob` vs
    # SPEECH_THRESHOLD, detect the *transition*:
    #   - if we were NOT speaking and now prob is above threshold -> print
    #     "speech started" and update state
    #   - if we WERE speaking and now prob is below threshold -> print
    #     "speech stopped" and update state
    #   - otherwise, do nothing (avoid spamming prints every 32ms)
    if not state["is_speaking"]:
        if speech_prob > SPEECH_THRESHOLD:
            print(f"speech started: {datetime.now()}")
            state["is_speaking"] = True
            state["silence_chunks"] = 0
    else:
        if speech_prob > SPEECH_THRESHOLD:
             state["silence_chunks"] = 0
        else:
            state["silence_chunks"] = state["silence_chunks"] + 1
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
    main()
