import subprocess, os, tempfile
import librosa
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr


def convert_to_wav(input_path):
    if input_path is None:
        return None
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    subprocess.run(
        ["ffmpeg", "-y", "-i", input_path, "-ar", "22050", "-ac", "1", tmp.name],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    return tmp.name


def load_audio(audio_path):
    wav_path = convert_to_wav(audio_path)
    audio, sr = librosa.load(wav_path, sr=None)
    return audio, sr


def calculate_spectral_features(audio, sr):
    S = np.abs(librosa.stft(audio))
    spectral_centroid = librosa.feature.spectral_centroid(S=S)[0]
    spectral_flatness = librosa.feature.spectral_flatness(S=S)[0]
    spectral_rolloff  = librosa.feature.spectral_rolloff(S=S)[0]
    mfcc      = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    return spectral_centroid, spectral_flatness, spectral_rolloff, mfcc_mean


def compare_audio_features(f1, f2):
    min_c = min(len(f1[0]), len(f2[0]))
    min_f = min(len(f1[1]), len(f2[1]))
    min_r = min(len(f1[2]), len(f2[2]))
    centroid_diff = np.mean(np.abs(f1[0][:min_c] - f2[0][:min_c]))
    flatness_diff = np.mean(np.abs(f1[1][:min_f] - f2[1][:min_f]))
    rolloff_diff  = np.mean(np.abs(f1[2][:min_r] - f2[2][:min_r]))
    cosine_sim    = np.dot(f1[3], f2[3]) / (np.linalg.norm(f1[3]) * np.linalg.norm(f2[3]))
    mfcc_distance = 1 - cosine_sim
    spectral_score = centroid_diff + flatness_diff + rolloff_diff
    return spectral_score, mfcc_distance, centroid_diff, flatness_diff, rolloff_diff


def get_verdict(mfcc_distance):
    if mfcc_distance < 0.10:
        return "✅ Same person.  (High confidence)"
    elif mfcc_distance < 0.25:
        return "⚠️ Uncertain — possibly different persons.  (Low confidence)"
    else:
        return "❌ Different persons.  (High confidence)"


def determine_similarity(audio_path_1, audio_path_2):
    audio_1, sr_1 = load_audio(audio_path_1)
    audio_2, sr_2 = load_audio(audio_path_2)
    f1 = calculate_spectral_features(audio_1, sr_1)
    f2 = calculate_spectral_features(audio_2, sr_2)
    result = compare_audio_features(f1, f2)
    mfcc_distance = result[1]
    verdict = get_verdict(mfcc_distance)
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
    return verdict, mfcc_distance, spectrogram_path


def gradio_interface(audio1, audio2):
    if audio1 is None or audio2 is None:
        return "Please upload both audio files.", None
    verdict, mfcc_dist, spectrogram_path = determine_similarity(audio1, audio2)
    result_text = (
        verdict + "\n" +
        "MFCC Distance: " + str(round(mfcc_dist, 4)) + "\n" +
        "Threshold: < 0.10 same | 0.10-0.25 uncertain | > 0.25 different"
    )
    return result_text, spectrogram_path


interface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Audio(type="filepath", label="Audio 1", format="wav"),
        gr.Audio(type="filepath", label="Audio 2", format="wav")
    ],
    outputs=[
        gr.Textbox(label="Result", lines=3),
        gr.Image(label="Spectrograms")
    ],
    title="Voice Similarity Checker",
    description="Upload two voice recordings to check if they belong to the same person. Supports WAV, MP3, OGG, OPUS, M4A, FLAC and more. Gradio converts all formats to WAV automatically.",
    examples=[["audio_path_1.wav", "audio_path_2.wav"]]
)

interface.launch()
