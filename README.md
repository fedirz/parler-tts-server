# Parler-TTS-Server
This repository provides a server with an [OpenAI compatible API](https://platform.openai.com/docs/api-reference/audio/createSpeech) interface for [Parler-TTS](https://github.com/huggingface/parler-tts).

## Quick Start 
Docker
```bash
docker run --detach --volume ~/.cache/huggingface:/root/.cache/huggingface --publish 8000:8000 fedirz/parler-tts-server
```
Using a fine-tuned model. See [main.py](./parler_tts_server/main.py) for configurable options
```bash
docker run --detach --volume ~/.cache/huggingface:/root/.cache/huggingface --publish 8000:8000 --env MODEL="parler-tts/parler-tts-mini-expresso" fedirz/parler-tts-server
```
Docker Compose
```bash
curl -sO https://raw.githubusercontent.com/fedirz/parler-tts-server/master/compose.yaml
docker compose up --detach parler-tts-server
```

## Usage 
Saving to file
```bash
curl -s -H "content-type: application/json" localhost:8000/v1/audio/speech -d '{"input": "Hey, how are you?"}' -o audio.mp3
```
Specifying a different format. WARN: there's a bug in the implementation causing audio to be distorted
```bash
curl -s -H "content-type: application/json" localhost:8000/v1/audio/speech -d '{"input": "Hey, how are you?", "response_type": "wav"}' -o audio.wav
```
Playing back the audio
```bash
curl -s -H "content-type: application/json" localhost:8000/v1/audio/speech -d '{"input": "Hey, how are you?"}' | ffplay -hide_banner -autoexit -nodisp -loglevel quiet -
```
Describing the voice the model should output
```bash
curl -s -H "content-type: application/json" localhost:8000/v1/audio/speech -d '{"input": "Hey, how are you?", "voice": "Feminine, speedy, and cheerfull"}' | ffplay -hide_banner -autoexit -nodisp -loglevel quiet -
```
Using `openai` should also work although haven't tested it yet - https://platform.openai.com/docs/guides/speech-to-text/speech-to-text
```python
from pathlib import Path
from openai import OpenAI
client = OpenAI()

speech_file_path = Path(__file__).parent / "speech.mp3"
response = client.audio.speech.create(
  model="parler-tts/parler_tts_mini_v0.1", # this is the name of the default model
  input="Today is a wonderful day to build something people love!"
)

response.stream_to_file(speech_file_path)
```
## Roadmap
- Add GitHub Actions for building and publishing the Docker image
- Provide a smaller Docker image for CPU only inference
- Add ARM support 
- Maybe: merge into [Parler-TTS](https://github.com/huggingface/parler-tts)

## Citations
```
@misc{lacombe-etal-2024-parler-tts,
  author = {Yoach Lacombe and Vaibhav Srivastav and Sanchit Gandhi},
  title = {Parler-TTS},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/huggingface/parler-tts}}
}
```
```
@misc{lyth2024natural,
      title={Natural language guidance of high-fidelity text-to-speech with synthetic annotations},
      author={Dan Lyth and Simon King},
      year={2024},
      eprint={2402.01912},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
}
```
