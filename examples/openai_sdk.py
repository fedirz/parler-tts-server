import os
from pathlib import Path

from openai import OpenAI

os.environ["OPENAI_API_KEY"] = "doesnt-matter-but-must-be-set"
os.environ["OPENAI_BASE_URL"] = "http://localhost:8000/v1"

client = OpenAI()

speech_file_path = Path(__file__).parent / "speech.mp3"
response = client.audio.speech.create(
    model="parler-tts/parler-tts-mini-expresso",
    input="Today is a wonderful day to build something people love!",
    voice="Thomas speaks moderately slowly in a sad tone with emphasis and high quality audio.",  # type: ignore
)

response.stream_to_file(speech_file_path)

# ffplay -hide_banner -autoexit -nodisp -loglevel quiet examples/speech.mp3
