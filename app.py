import subprocess, os, tempfile
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr

# Resemblyzer se instala en el Space via requirements.txt
from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path


encoder = VoiceEncoder()


def convert_to_wav(input_path):
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    subprocess.run(
        ["ffmpeg", "-y", "-i", input_path, "-ar", "16000", "-ac", "1", tmp.name],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    return tmp.name


def get_embedding(audio_path):
    wav_path = convert_to_wav(audio_path)
    wav = preprocess_wav(Path(wav_path))
    embedding = encoder.embed_utterance(wav)
    return embedding


def get_verdict(similarity):
    if similarity > 0.85:
        return "✅ Same person.  (High confidence)"
    elif similarity > 0.70:
        return "⚠️ Possibly same person.  (Low confidence)"
    else:
        return "❌ Different persons.  (High confidence)"


def plot_spectrograms(path1, path2):
    import librosa
    audio_1, sr_1 = librosa.load(path1, sr=None)
    audio_2, sr_2 = librosa.load(path2, sr=None)
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    librosa.display.specshow(
        librosa.amplitude_to_db(np.abs(librosa.stft(audio_1)), ref=np.max),
        sr=sr_1, x_axis="time", y_axis="log", ax=axes[0])
    axes[0].set_title("Spectrogram - Audio 1")
    fig.colorbar(axes[0].collections[0], ax=axes[0], format="%+2.0f dB")
    librosa.display.specshow(
        librosa.amplitude_to_db(np.abs(librosa.stft(audio_2)), ref=np.max),
        sr=sr_2, x_axis="time", y_axis="log", ax=axes[1])
    axes[1].set_title("Spectrogram - Audio 2")
    fig.colorbar(axes[1].collections[0], ax=axes[1], format="%+2.0f dB")
    plt.tight_layout()
    spectrogram_path = "/tmp/spectrogram_output.png"
    plt.savefig(spectrogram_path, dpi=150, bbox_inches="tight")
    plt.close()
    return spectrogram_path


def gradio_interface(file1, file2):
    if file1 is None or file2 is None:
        return "Please upload both audio files.", None
    path1 = file1.name if hasattr(file1, "name") else file1
    path2 = file2.name if hasattr(file2, "name") else file2
    emb1 = get_embedding(path1)
    emb2 = get_embedding(path2)
    similarity = float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
    verdict = get_verdict(similarity)
    spec_path = plot_spectrograms(convert_to_wav(path1), convert_to_wav(path2))
    result_text = (
        verdict + "\n" +
        "Similarity score: " + str(round(similarity, 4)) + "\n" +
        "Threshold: > 0.85 same | 0.70-0.85 uncertain | < 0.70 different"
    )
    return result_text, spec_path


interface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.File(label="Audio 1 (WAV, MP3, OGG, OPUS, M4A, FLAC...)"),
        gr.File(label="Audio 2 (WAV, MP3, OGG, OPUS, M4A, FLAC...)")
    ],
    outputs=[
        gr.Textbox(label="Result", lines=3),
        gr.Image(label="Spectrograms")
    ],
    title="Voice Similarity Checker",
    description="Upload two voice recordings to check if they belong to the same person. Uses Resemblyzer (Google GE2E) — a deep learning voice encoder trained on thousands of speakers."
)

interface.launch()
