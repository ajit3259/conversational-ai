# Stage 1 · Mic capture

Opens the default input device and prints the loudness of each incoming
block.

## Run

```bash
uv run stage1_mic_capture/capture.py
```

Stop with Ctrl+C.

## What to expect

A continuous stream of readings, roughly 33 per second:

```
rms 0.0004   dBFS  -67.9
rms 0.0381   dBFS  -28.4
```

Talk and the numbers jump. Stay quiet and they settle near zero — but they
keep arriving either way, which is the thing worth noticing. Audio is
delivered on the sound card's clock, not when there is something to hear.
Silence is not an absence of data, it is blocks of numbers near zero.

dBFS is measured downward from the loudest representable signal, so real
audio always reads negative. Zero would mean clipping.

## What it cannot do

Tell speech apart from a door closing, or know when a sentence ended. That
is [stage 2](../stage2_vad).
