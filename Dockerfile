FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime
RUN apt-get update && apt-get install -y --no-install-recommends git
WORKDIR /root/parler-tts-server
RUN pip install --no-cache-dir --no-deps git+https://github.com/huggingface/parler-tts.git 
COPY ./model_requirements.txt .
RUN pip install --no-cache-dir -r model_requirements.txt
COPY ./server_requirements.txt .
RUN pip install --no-cache-dir -r server_requirements.txt
COPY ./parler_tts_server ./parler_tts_server
CMD ["uvicorn", "parler_tts_server.main:app"]
ENV UVICORN_HOST=0.0.0.0
ENV UVICORN_PORT=8000
