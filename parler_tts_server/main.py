import logging
import time
from contextlib import asynccontextmanager
from typing import Annotated

import soundfile as sf
import torch
from fastapi import Body, FastAPI, Response
from fastapi.responses import FileResponse
from parler_tts import ParlerTTSForConditionalGeneration
from pydantic_settings import BaseSettings
from transformers import AutoTokenizer

MODEL = "parler-tts/parler_tts_mini_v0.1"
VOICE = "A female speaker with a slightly low-pitched voice delivers her words quite expressively, in a very confined sounding environment with clear audio quality. She speaks very fast."
RESPONSE_FORMAT = "mp3"
SPEED = 1.0


class Config(BaseSettings):
    log_level: str = "info"  # env: LOG_LEVEL
    model: str = MODEL  # env: MODEL
    voice: str = VOICE  # env: VOICE
    response_format: str = RESPONSE_FORMAT  # env: RESPONSE_FORMAT


config = Config()
root_logger = logging.getLogger()
root_logger.setLevel(logging.CRITICAL)
logger = logging.getLogger(__name__)
logger.setLevel(config.log_level.upper())

# https://github.com/huggingface/parler-tts?tab=readme-ov-file#usage
if torch.cuda.is_available():
    device = "cuda:0"
    logging.info("GPU will be used for inference")
else:
    device = "cpu"
    logging.info("CPU will be used for inference")
torch_dtype = torch.float16 if device != "cpu" else torch.float32

tts: ParlerTTSForConditionalGeneration = None  # type: ignore


@asynccontextmanager
async def lifespan(_: FastAPI):
    global tts
    torch_dtype = torch.float16 if device != "cpu" else torch.float32
    logging.debug(f"Loading {config.model}")
    start = time.perf_counter()
    tts = ParlerTTSForConditionalGeneration.from_pretrained(
        config.model,
    ).to(  # type: ignore
        device, dtype=torch_dtype  # type: ignore
    )
    logger.info(
        f"Loaded {config.model} loaded in {time.perf_counter() - start:.2f} seconds"
    )
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health() -> Response:
    return Response(status_code=200, content="OK")


# https://platform.openai.com/docs/api-reference/audio/createSpeech
@app.post("/v1/audio/speech")
async def generate_audio(
    input: Annotated[str, Body()],
    voice: Annotated[str, Body()] = config.voice,
    model: Annotated[str, Body()] = config.model,
    response_format: Annotated[str, Body()] = config.response_format,
    speed: Annotated[float, Body()] = SPEED,
):
    if model != config.model:
        logger.warning(
            f"Specifying a model that is different from the default is not supported yet. Using default model: {config.model}."
        )
    if speed != SPEED:
        logger.warning(
            f"Specifying speed isn't supported by {config.model}. Using default speed: {SPEED}."
        )
    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(config.model)
    input_ids = tokenizer(voice, return_tensors="pt").input_ids.to(device)
    prompt_input_ids = tokenizer(input, return_tensors="pt").input_ids.to(device)
    generation = tts.generate(
        input_ids=input_ids, prompt_input_ids=prompt_input_ids
    ).to(  # type: ignore
        torch.float32
    )
    audio_arr = generation.cpu().numpy().squeeze()
    logger.info(
        f"Took {time.perf_counter() - start:.2f} seconds to generate audio for {len(input.split())} words using {device.upper()}"
    )
    # TODO: use an in-memory file instead of writing to disk
    sf.write(f"out.{response_format}", audio_arr, tts.config.sampling_rate)
    return FileResponse(f"out.{response_format}", media_type=f"audio/{response_format}")
