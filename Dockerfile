FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements_space.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY audio_path_1.wav .
COPY audio_path_2.wav .

EXPOSE 7860

CMD ["python", "app.py"]
