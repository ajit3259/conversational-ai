# Doubt Pad

Running log of questions/doubts raised while reviewing each stage.
Some are answered inline; some are deliberately left open to be
self-answered while working through later stages.

## Glossary (new terms flagged for later, deeper look)

- **Diarization**: figuring out "who spoke when" in multi-speaker audio
  (segmenting/labeling turns by speaker identity), e.g. meeting
  transcription tools.
- **PCM** (Pulse-Code Modulation): the standard uncompressed
  representation of digital audio as a raw sequence of amplitude
  samples. We've been working with PCM the whole time — our float32
  arrays are floating-point PCM, vs. the more common integer PCM (e.g.
  16-bit PCM in standard `.wav` files).
- **Aliasing**: when a signal is sampled/resampled too slowly to
  capture its higher frequencies, those frequencies get mathematically
  disguised as false, incorrect lower frequencies in the result (not
  just lost — actively wrong). Ties to the Nyquist rule (Stage 1): need
  >= 2x sample rate vs. the highest frequency you want to represent.
  Why downsampling needs a low-pass filter first, to remove frequencies
  that would exceed the new Nyquist limit before they can fold back
  down and corrupt the signal as audible distortion.

## Stage 6: Orchestration -- production readiness addendum

(Most Stage 6 doubts were covered organically while debugging the
feedback loop and barge-in race conditions. Closing notes on what
"production ready" adds beyond this build:)

- **Concurrency architecture**: our `main()` is a single blocking loop,
  one utterance at a time. Real service needs many simultaneous
  users -- `asyncio` with one task per session, or a worker pool, not
  our simple synchronous loop.
- **Client/server split**: our mic/speakers are local to the same
  process as the models. Real products stream audio from a remote
  client (browser/phone) over WebSocket to a server running the heavy
  models -- network jitter, packet loss, codec negotiation all become
  real concerns not present in a local script.
- **Resilience**: no retry/fallback logic currently -- a crash in
  `requests.post()` just kills the whole session. Real systems need
  timeouts, retries, graceful degradation.
- **Observability**: logging + per-stage latency metrics
  (time-to-first-token, time-to-first-audio, ASR/TTS duration) --
  can't just read print statements in production.
- **Resource management**: model loading is expensive (felt directly
  waiting for Whisper/Silero/Piper to load) -- production keeps models
  warm across requests/sessions, needs real GPU capacity planning for
  concurrent users.
- **Stateful sessions scale differently than stateless HTTP**: a voice
  session lives for minutes, holds conversation history + audio
  buffers in memory -- complicates horizontal scaling/load balancing
  compared to typical stateless web request scaling.

## Stage 1: Mic capture

**Understanding check (confirmed correct):** Sample rate = how many times
per second amplitude is measured. 16kHz = 16000 measurements/sec, each a
float. Positive/negative sign is just the direction of pulsation (above
or below the resting baseline), not "loud vs quiet."

**Q: Is `sounddevice` a Python standard library / used in production?**
A: No — it's a third-party package (`pip install sounddevice`), a thin
wrapper around the C library PortAudio. Not stdlib, but PortAudio itself
is legitimately used in production audio software. Whether *this exact
Python wrapper* is what a large-scale production system would use is a
separate question (see the threading doubt below).

**Q: Does `InputStream` always fire the callback at a fixed block size,
regardless of silence?**
A: Yes. The callback fires purely based on the audio hardware clock
producing `BLOCK_SIZE` new samples — it has no concept of "silence" or
"content." Silence is just a chunk full of near-zero-amplitude samples;
VAD (Stage 2) is what adds the concept of "is this speech or not" on top
of these otherwise-identical fixed-size chunks.

**Q: How does "volume" connect to RMS?**
A: RMS (root-mean-square) *is* the volume measurement we used. A single
raw sample doesn't tell you "loudness" (it's just one instantaneous
amplitude value, and could be positive or negative). RMS squares every
sample in a chunk (removing the sign), averages them, then square-roots
back to the original scale — giving one number per chunk that represents
the chunk's overall energy/loudness.

**Q: Why is dB always negative?**
A: dBFS ("decibels relative to Full Scale") is a *ratio* to the loudest
possible signal your system can represent (amplitude 1.0). `0 dBFS`
means "as loud as it's possible to get without clipping." Since almost
all real audio sits below that maximum, `20 * log10(rms)` comes out
negative — the more negative, the quieter. It only approaches 0 right at
the edge of distortion/clipping. (Separately: near-total silence produces
wild, meaningless dB swings due to `log10` of near-zero floating point
noise — the clamping issue we noted at the time.)

**Q: What does "clipping" mean?**
A: When the true sound is louder than the max value the system can
represent (`1.0` for our float32 samples), the peaks get clamped/cut off
flat at that max instead of recording the real waveform shape — heard as
harsh, crackly distortion. `0 dBFS` is that clipping ceiling, which is why
good recording levels stay well below it, not right at it.

**Open doubt (self-answer as you go further):** the `while True: sd.sleep()`
loop is clearly not how a real multi-user production service would work —
you suspect each user connection would get its own thread/session,
torn down on disconnect rather than relying on Ctrl+C. Correct instinct;
worth revisiting concretely once we look at how Stage 6 would need to
change to serve multiple simultaneous users (this single-process,
single-user script is a deliberate simplification for learning, not a
production architecture).

## Stage 2: VAD

**Understanding check (confirmed correct):** VAD decides whether mic
input is silence/noise or actual human speech — either via a heuristic
or an AI model — so we don't waste downstream processing (or produce
output) in response to non-speech.

**Q: Important VAD models?**
A: WebRTC VAD (classic, statistical/energy-based, very lightweight),
Silero VAD (what we used — small neural net, good CPU speed/accuracy
tradeoff), pyannote VAD (heavier, more accurate, common in diarization),
NVIDIA MarbleNet/NeMo, Picovoice Cobra (commercial).

**Q: Why threshold = 0.5, what's the intuition?**
A: No deep derivation — it's the natural "more likely speech than not"
midpoint of a 0-1 probability, and Silero's own suggested default. Real
tuning is an empirical precision/recall tradeoff (raise = fewer false
triggers but misses quiet speech; lower = catches quiet speech but more
false positives), tuned against labeled data in production, not derived
theoretically.

**Q: Besides sample rate, what other pre-processing might be needed?**
A: Channel count (mono vs stereo — models expect mono, need downmixing
if not), sample format/dtype (e.g. int16 PCM -> float32 in [-1,1]),
matching the model's expected chunk/frame size, sometimes
normalization for wildly varying input levels. Classical (non-neural)
pipelines also used techniques like pre-emphasis filtering; end-to-end
neural models like Silero skip that, taking raw waveform directly.

**Q: Production state — per-user in-memory, main thread hands off to a
user thread?**
A: Yes, essentially right — our single global `state` dict only works
because this is a single-user demo. Real multi-user service = fresh
state object per connection/session, never shared. Nuance: at scale,
"thread per user" might actually be a lighter-weight async task/session
object rather than a full OS thread, since OS threads are relatively
heavy — but "isolated state per session" is the correct principle
regardless of the concurrency primitive used.

**Q: VAD "families" like TTS's LM-based/diffusion/flow?**
A: Different category entirely — those TTS terms are *generative* model
families (TTS produces audio). VAD is a *discriminative classifier*
(speech vs not), so it doesn't share that taxonomy. VAD's real
"families" are about classifier architecture: rule-based/statistical
(WebRTC), classical ML on hand-crafted features (older systems), or
small neural nets (Silero).

**Q: Can VAD be made smarter about abrupt noise/bursts vs real speech?**
A: Yes — real techniques include: dedicated acoustic event detection
models alongside VAD (flagging coughs/claps/clicks specifically);
cross-checking against ASR confidence (VAD fired but ASR output is
garbage -> treat as false trigger); multi-class VAD (speech /
human-non-speech / noise / silence) instead of binary; directional
filtering via mic arrays. There's also newer "semantic turn detection"
research using small LMs on partial transcripts instead of pure
acoustic silence timing (relevant to our deferred adaptive-hangover
topic).

**Q: What does "channel" mean and how does it affect VAD?**
A: A channel = one continuous stream of samples from one audio source
(mono = 1, stereo = 2). VAD models expect mono; multi-channel input
needs downmixing first (we avoided this by recording `channels=1` from
Stage 1 onward). Multiple channels matter for mic *arrays* — multiple
physical mics enable beamforming (estimating sound direction) to focus
on a speaker and suppress background noise/echo from other directions,
part of how real smart speakers handle noisy rooms — ties back to the
AEC discussion as a complementary hardware-side technique.

## Stage 3: ASR

**Correction (not a doubt, a misconception to fix):** ASR and TTS are
*not* the same — they're inverses. ASR = speech -> text (Stage 3,
Whisper). TTS = text -> speech (Stage 5, Piper). Opposite directions,
different model architectures, even though built as a pair here.

**Q: Streaming vs. non-streaming ASR, how do they differ?**
A: Non-streaming (what we built) waits for a complete audio segment
(VAD tells us when it's done), then transcribes once. Streaming ASR
continuously emits *partial* hypotheses as audio arrives, revising them
as more audio comes in (like live captions updating word-by-word).
What we built is "segment-then-transcribe" — a simpler approximation,
not true low-latency streaming.

**Q: ASR model families?**
A: Whisper (encoder-decoder Transformer, attention over the whole
segment — naturally suited to offline/batch, not true streaming).
CTC-based (wav2vec2, older DeepSpeech — per-frame token probabilities).
RNN-Transducer (RNN-T) — the architecture built specifically for real
streaming (what Google's on-device/real-time ASR uses), since it
processes audio incrementally by design.

**Q: Should real-time always involve true streaming?**
A: For lowest latency, RNN-T-style streaming is the "correct"
architecture. But many practical real-time products (including many
voice assistants) use segment-then-transcribe like we did — a genuine
engineering tradeoff (simplicity/latency budget), not a strict
requirement.

**Q: Input restrictions on ASR?**
A: Sample rate (16kHz for Whisper), max audio length per inference call
(Whisper processes internally in 30s windows, longer audio needs
chunking), mono channel expected, and language/accent coverage entirely
dependent on training data (part of why "Ajit" -> "Adith" happened).

**Q: Can ASR identify expressions/emotion, not just words?**
A: No, not standard ASR — just words + basic punctuation. That's a
separate task, speech emotion recognition (SER) / paralinguistics,
using different models trained on tone/prosody. Newer "audio-native"
models (e.g. GPT-4o voice mode, processing audio directly rather than
ASR-then-LLM) can jointly reason about tone and words — a genuinely
different architecture than our ASR -> LLM -> TTS pipeline.

**Q: Is batching across requests done for real-time ASR?**
A: Yes — production serving (NVIDIA Riva, Triton Inference Server) does
dynamic/continuous batching: grouping multiple concurrent users' audio
into one GPU forward pass for throughput, same core idea as LLM
serving. Trades a small added latency (waiting to accumulate a batch)
for much better GPU utilization at scale.

## Stage 4: LLM dialogue

**Insight (confirmed correct, with nuance):** Speculative prefill —
start the LLM's prefill phase (building the KV cache from a partial
ASR transcript) before the user finishes talking, so only the faster
decode phase remains once VAD confirms end-of-turn, then start
streaming to TTS as soon as decode begins. Real technique, genuinely
used for cutting time-to-first-audio. Requires *streaming* ASR with
partial hypotheses (which we didn't build — ours only emits final
transcripts), and has real complexity if the final sentence changes
from what was speculatively prefilled (wasted/invalid cache).

**Confirmed:** any capable text LLM can slot into the dialogue stage —
this is why swapping LM Studio -> Ollama only touched Stage 4's file.
Most swappable/model-agnostic stage in a cascaded pipeline.

**Confirmed:** tool calls add real latency (orchestrator intercepts,
calls the tool, feeds result back to the LLM before anything reaches
TTS). Filler/stalling responses ("Let me check that for you...") played
immediately via TTS while the tool call runs in the background is a
real, widely-used production technique to mask (not eliminate) that
latency.

**Confirmed, with real metric names:** "low latency + good quality" ->
**time-to-first-audio** (latency), **WER** (word error rate, ASR
quality), **MOS** (mean opinion score, TTS naturalness), response
coherence (LLM quality).

**Confirmed:** for a "final version," real orchestrator frameworks
exist instead of hand-rolling the plumbing: **LiveKit Agents**,
**Pipecat**, **Vocode** (open-source), OpenAI's Realtime API and
similar (hosted). Building it by hand here was right for learning.

**Q: Cascaded pipeline (what we built) vs. omni/native voice-to-voice
-- what are the real tradeoffs?**
A: What we built is called a **cascaded** architecture (ASR -> LLM ->
TTS, text as the bottleneck between stages, everything independently
swappable). The alternative (e.g. GPT-4o voice mode, Kyutai's Moshi) is
**native/omni speech-to-speech** -- one model, audio in, audio out
directly.
- Cascaded: lower engineering risk, every stage swappable/debuggable
  (proven by our LLM backend swap), tool-calling is straightforward
  (text-mediated), but adds latency (multiple hops) and **loses
  paralinguistic info** -- the LLM only ever sees *what* was said as
  text, never tone/emphasis/sighs/laughs/"how" it was said.
- Native/omni: much lower latency (no hop-to-hop serialization), can
  preserve/generate genuine expressiveness since it never collapses
  through a lossy text bottleneck -- but much harder to train (needs
  massive paired audio+text data), harder to add explicit tool-calling
  cleanly, and far less swappable (can't independently upgrade "just
  the ASR part" once it's fused into one model).
- Middle ground: audio-native understanding (no separate classical ASR
  stage) combined with a still-separate TTS stage -- cuts one hop of
  latency/information loss while keeping voice output swappable.

## Stage 5: TTS

**Insight (confirmed correct):** every stage is heavily
language/accent-specific -- felt directly via separate en_US vs hi_IN
voice downloads, and ASR accuracy tied to training data's
accent/language coverage. Related hard problem: code-switching (mixing
languages mid-sentence, e.g. Hinglish) -- most ASR/TTS models are
trained largely monolingual and handle this poorly. Known, unsolved-at-
scale industry problem, not specific to this toy build.

**Insight (confirmed correct):** cascaded pipelines compound errors
with no self-correction by default -- ASR mistake -> LLM reasons about
wrong input -> TTS speaks the wrong output confidently, no feedback
loop. Real mitigations that exist: passing ASR confidence/N-best
hypotheses to the LLM, confirmation turns for high-stakes actions
("You said transfer $500 -- correct?"), LLMs' natural robustness to
slightly noisy text. But nothing automatic unless deliberately
designed in.

**Insight (Indic languages hardest + market economics):** technical
part solid -- less training data, 22+ official languages, huge
dialectal variation, plus code-switching. The labor-cost economic
causality claim is a reasonable hypothesis, not something I can
confirm/refute with hard data.

**Q: Voice/sample rate is model-dependent, but can we always
post-process to our own target rate?**
A: Yes -- resampling (`librosa.resample`, `scipy.signal.resample`,
`soxr`) converts a model's native output rate to any target (e.g.
matching an 8kHz telephony codec). Limits: upsampling doesn't add real
fidelity (can't manufacture detail that wasn't generated), downsampling
needs proper anti-aliasing filtering or introduces artifacts. Possible,
not entirely free.

**Q/Catch (confirmed real gap in our own build): are we actually
streaming LLM output straight into TTS, or buffering then streaming?**
A: Buffering then streaming -- a real gap we built but never closed.
`stream_chat()` prints tokens live but only *returns* `full_text` after
the entire response finishes generating; `speak()` is only called with
that complete text afterward. Piper's internal per-sentence chunking
only kicks in after the full LLM response already exists -- the two
streaming stages (Stage 4 LLM streaming, Stage 5 TTS sentence
streaming) were demonstrated independently but never actually wired to
overlap. Real fix: detect a completed sentence *as LLM tokens stream
in* (watching for `.`/`!`/`?`) and hand it to TTS immediately while the
LLM keeps generating the next sentence -- a concrete latency
improvement still on the table for this build.

## Depth topic: AEC live integration -- concluding limitations

Built and validated NLMS + Geigel double-talk detection from scratch
(`depth_aec/`), then integrated into the live pipeline as
`stage6_orchestration/assistant_aec.py` (a duplex `sd.Stream`, since
AEC needs sample-aligned reference+mic audio -- a real architectural
upgrade over Pass A/B's two separate streams).

Real self-triggering (barge-in on the assistant's own voice) was
genuinely fixed. Along the way, found and fixed real bugs: filter
length far too short for a real room's echo tail (200 samples/12.5ms
-> 1600/100ms), a "skip AEC when nothing's playing" optimization,
and a classic Python performance anti-pattern (popping from the front
of a list is O(n); switched to `collections.deque` for O(1)).

**Final unresolved limitation**: persistent audible noise during live
playback, isolated (via a debug WAV recording of the exact `outdata`
samples) to be a hardware/driver-level real-time duplex issue, not a
bug in our algorithm or data -- the recorded "intended" audio played
back perfectly clean. `latency="high"` didn't fully resolve it.
Insight: NLMS assumes a *linear* echo path; if the hardware's own
duplex handling introduces nonlinear glitching into actual playback,
AEC structurally can't predict/cancel that distortion, so it leaks
through as residual and can re-trigger false barge-in -- same root
cause (hardware duplex limitation) producing both symptoms.

Concluded here as a complete, honest result rather than chasing a
hardware/driver limitation further.

**WebRTC AEC3 attempt**: tried `aec-audio-processing` (pip package
wrapping WebRTC's Audio Processing Module). Build failed even after
installing `ninja`/`meson`: `error: no template named 'Nullable' in
namespace 'absl'` -- the vendored WebRTC source expects a different
Abseil API version than the Homebrew-installed one provides. Genuine
native C++ dependency version mismatch, not a bug in our approach.
Declined to chase pinning an older Abseil -- concluded as a real,
valid lesson: "just use an existing library" has real friction too,
especially for something like WebRTC that vendors a huge C++ codebase
with its own build requirements.
