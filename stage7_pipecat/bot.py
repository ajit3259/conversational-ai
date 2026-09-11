"""
Stage 7: the same assistant, assembled rather than built.

Everything stages 1-6 worked through by hand, using the same four models --
Silero, Whisper, a local Gemma through Ollama, Piper -- wired together by
Pipecat instead of by us.

What the framework is doing that we had to do ourselves:

  - The browser captures audio, which means echo cancellation arrives free
    from getUserMedia. This is the wall stage 6 pass C ran into, and it is
    gone here because we stopped owning the microphone.
  - Turn-taking, interruption and session state are handled. Note where the
    VAD is attached below: to the *user context aggregator*, not to the
    transport. The framework treats deciding whose turn it is as part of the
    conversation rather than part of the audio.
  - LLM output streams into TTS sentence by sentence, which is the seam our
    own build never closed.
  - Resampling between the models' differing rates is somebody else's
    problem now.

Nothing here is cleverer than what we wrote. It is the same pipeline with
the joints already solved, which was the point of taking it apart first.

Run with:  uv run bot.py
Then open http://localhost:7860 and allow microphone access.
"""

from pathlib import Path

from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import WorkerRunner
from pipecat.pipeline.task import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.ollama.llm import OLLamaLLMService
from pipecat.services.piper.tts import PiperTTSService
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.transports.base_transport import BaseTransport, TransportParams

# Reuse the voice already downloaded for stage 5. Piper fetches it here if it
# is missing.
VOICES_DIR = Path(__file__).resolve().parent.parent / "stage5_tts" / "voices"
VOICE_ID = "en_US-lessac-medium"

MODEL = "gemma3n:e4b"

# The same constraint as stage 4, and for the same reason: this text is going
# to be spoken, not displayed.
SYSTEM_PROMPT = (
    "You are a helpful voice assistant. Answer in 1-3 short spoken "
    "sentences. Do not use markdown, bullet points, asterisks, or emoji, "
    "since your response will be read aloud by a text-to-speech system."
)

transport_params = {
    "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments) -> None:
    # The same three models from stages 3, 4 and 5, now as services.
    stt = WhisperSTTService(model="small", device="cpu", compute_type="int8")
    llm = OLLamaLLMService(model=MODEL)
    tts = PiperTTSService(voice_id=VOICE_ID, download_dir=VOICES_DIR)

    context = LLMContext([{"role": "system", "content": SYSTEM_PROMPT}])

    # Silero again, but notice where it goes. In stage 6 the VAD lived in the
    # audio callback, next to the microphone. Here it is a property of the
    # thing that decides when a user's turn is complete.
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    # Stage 6's main loop, as a list.
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    worker = PipelineWorker(pipeline, params=PipelineParams(enable_metrics=True))
    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)

    # Pipecat calls these positionally, so the arguments are named out of the
    # way here rather than shadowing the transport we are attaching them to.
    @transport.event_handler("on_client_connected")
    async def on_client_connected(_transport, _client):
        logger.info("Client connected")
        context.add_message(
            {"role": "developer", "content": "Start by concisely introducing yourself."}
        )
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(_transport, _client):
        logger.info("Client disconnected")
        await runner.cancel()

    await runner.run()


async def bot(runner_args: RunnerArguments) -> None:
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
