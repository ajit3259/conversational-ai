# Depth topic · Acoustic echo cancellation

Three offline experiments building up the filter that
[stage 6 pass C](../stage6_orchestration) uses live. Offline first, because
verifying a canceller against a room you do not control tells you nothing.

Run from the repo root — the scripts load their cached audio by relative
path.

## 1. Cancel a known echo

```bash
uv run depth_aec/nlms_experiment.py
```

The echo path is fabricated here: a fixed delay and attenuation applied to a
synthesized sentence. If the filter cannot cancel an echo we invented
ourselves, nothing else is worth trying.

Prints ERLE in dB — how much echo energy went away, higher being better —
then plays the mic signal and the cleaned residual so you can hear the
difference.

Needs a voice model, since it synthesizes its reference:

```bash
uv run python -m piper.download_voices en_US-lessac-medium --download-dir stage5_tts/voices
```

## 2. Does genuine speech survive?

```bash
uv run depth_aec/nlms_experiment_v2.py
```

Adds an independent near-end voice — a completely different sentence — on
top of the echo, which is the case that actually matters. Cancelling echo is
useless if it cancels the user too.

Prints ERLE plus the correlation between the residual and the original
voice. Close to 1 means the speech came through intact.

This is where the **double-talk problem** shows up: a loud independent voice
corrupts the filter's adaptation, because the filter tries to explain that
voice as echo.

Audio here is loaded from `reference.npy` and `near_end_voice.npy` rather
than synthesized. Piper is not deterministic across calls, so caching keeps
runs comparable — otherwise a change in the numbers might just be different
input audio.

## 3. Freeze adaptation during double-talk

```bash
uv run depth_aec/nlms_experiment_v3_doubletalk.py
```

Adds a Geigel detector: if the mic sample is louder than the recent
reference could plausibly explain as echo, somebody else is talking, so stop
learning while continuing to cancel with the weights already held.

Also reports what fraction of samples had adaptation frozen.

Expect correlation to improve sharply and ERLE to get slightly worse. That
trade is the right way round and it is what real systems do — protecting the
user's speech matters more than maximising echo suppression while both
parties are talking.

## Where this stops

Integrated into the live pipeline it did fix the self-triggering. It did not
fix a persistent crackle in duplex playback, which turned out to be the
hardware rather than the algorithm — and a filter of this kind assumes a
*linear* echo path, so distortion the hardware introduces is not a scaled
copy of anything and cannot be predicted or subtracted.

WebRTC's audio processing module is the obvious thing to compare against.
It failed to build here over a C++ dependency version mismatch, which is its
own kind of lesson about reaching for the standard implementation.
