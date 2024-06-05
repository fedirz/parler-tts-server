import enum
import time
from contextlib import asynccontextmanager
from typing import Annotated, Any, OrderedDict

import huggingface_hub
import soundfile as sf
import torch
from fastapi import Body, FastAPI, HTTPException, Response
from fastapi.responses import FileResponse
from huggingface_hub.hf_api import ModelInfo
from openai.types import Model
from parler_tts import ParlerTTSForConditionalGeneration
from pydantic_settings import BaseSettings
from transformers import AutoTokenizer

from parler_tts_server.logger import logger

SPEED = 1.0


# NOTE: commented out response formats don't work
class ResponseFormat(enum.StrEnum):
    MP3 = "mp3"
    # OPUS = "opus"
    # AAC = "aac"
    FLAC = "flac"
    WAV = "wav"
    # PCM = "pcm"


class Config(BaseSettings):
    log_level: str = "info"  # env: LOG_LEVEL
    model: str = "parler-tts/parler-tts-mini-expresso"  # env: MODEL
    max_models: int = 1  # env: MAX_MODELS
    lazy_load_model: bool = False  # env: LAZY_LOAD_MODEL
    voice: str = "Thomas speaks moderately slowly in a sad tone with emphasis and high quality audio."  # env: VOICE
    response_format: ResponseFormat = ResponseFormat.MP3  # env: RESPONSE_FORMAT


config = Config()

# https://github.com/huggingface/parler-tts?tab=readme-ov-file#usage
if torch.cuda.is_available():
    device = "cuda:0"
    logger.info("GPU will be used for inference")
else:
    device = "cpu"
    logger.info("CPU will be used for inference")
torch_dtype = torch.float16 if device != "cpu" else torch.float32


class ModelManager:
    def __init__(self):
        self.model_tokenizer: OrderedDict[
            str, tuple[ParlerTTSForConditionalGeneration, AutoTokenizer]
        ] = OrderedDict()

    def load_model(
        self, model_name: str
    ) -> tuple[ParlerTTSForConditionalGeneration, AutoTokenizer]:
        logger.debug(f"Loading {model_name}...")
        start = time.perf_counter()
        model = ParlerTTSForConditionalGeneration.from_pretrained(model_name).to(  # type: ignore
            device,  # type: ignore
            dtype=torch_dtype,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info(
            f"Loaded {model_name} and tokenizer in {time.perf_counter() - start:.2f} seconds"
        )
        return model, tokenizer

    def get_or_load_model(
        self, model_name: str
    ) -> tuple[ParlerTTSForConditionalGeneration, Any]:
        if model_name not in self.model_tokenizer:
            logger.info(f"Model {model_name} isn't already loaded")
            if len(self.model_tokenizer) == config.max_models:
                logger.info("Unloading the oldest loaded model")
                del self.model_tokenizer[next(iter(self.model_tokenizer))]
            self.model_tokenizer[model_name] = self.load_model(model_name)
        return self.model_tokenizer[model_name]


model_manager = ModelManager()


@asynccontextmanager
async def lifespan(_: FastAPI):
    if not config.lazy_load_model:
        model_manager.get_or_load_model(config.model)
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health() -> Response:
    return Response(status_code=200, content="OK")


@app.get("/v1/models", response_model=list[Model])
def get_models() -> list[Model]:
    models = list(huggingface_hub.list_models(model_name="parler-tts"))
    models = [
        Model(
            id=model.id,
            created=int(model.created_at.timestamp()),
            object="model",
            owned_by=model.id.split("/")[0],
        )
        for model in models
        if model.created_at is not None
    ]
    return models


@app.get("/v1/models/{model_name:path}", response_model=Model)
def get_model(model_name: str) -> Model:
    models = list(huggingface_hub.list_models(model_name=model_name))
    if len(models) == 0:
        raise HTTPException(status_code=404, detail="Model doesn't exists")
    exact_match: ModelInfo | None = None
    for model in models:
        if model.id == model_name:
            exact_match = model
            break
    if exact_match is None:
        raise HTTPException(
            status_code=404,
            detail=f"Model doesn't exists. Possible matches: {", ".join([model.id for model in models])}",
        )
    assert exact_match.created_at is not None
    return Model(
        id=exact_match.id,
        created=int(exact_match.created_at.timestamp()),
        object="model",
        owned_by=exact_match.id.split("/")[0],
    )


# https://platform.openai.com/docs/api-reference/audio/createSpeech
@app.post("/v1/audio/speech")
async def generate_audio(
    input: Annotated[str, Body()],
    voice: Annotated[str, Body()] = config.voice,
    model: Annotated[str, Body()] = config.model,
    response_format: Annotated[ResponseFormat, Body()] = config.response_format,
    speed: Annotated[float, Body()] = SPEED,
) -> FileResponse:
    tts, tokenizer = model_manager.get_or_load_model(model)
    if speed != SPEED:
        logger.warning(
            "Specifying speed isn't supported by this model. Audio will be generated with the default speed"
        )
    start = time.perf_counter()
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
