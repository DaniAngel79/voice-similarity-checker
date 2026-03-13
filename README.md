# Voice Similarity Checker

Audio biometrics tool that determines whether two voice recordings belong to the same person, using **MFCC cosine distance** and **spectral feature analysis**.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![Librosa](https://img.shields.io/badge/librosa-0.10%2B-green) ![Gradio](https://img.shields.io/badge/gradio-4.0%2B-orange)

---

## How it works

1. Loads two audio files (WAV or MP3) via `librosa`
2. Extracts **13 MFCC coefficients** per audio (mean over time)
3. Computes **cosine distance** between MFCC vectors
4. Extracts spectral features: centroid, flatness, rolloff
5. **Verdict:** MFCC distance < 0.15 → same person
6. Outputs a **spectrogram image** and a **PDF report**

## Technical stack

| Component | Library |
|-----------|---------|
| Audio loading | `librosa` |
| MFCC extraction | `librosa.feature.mfcc` |
| Similarity metric | Cosine distance (1 − cosine similarity) |
| Spectral features | centroid, flatness, rolloff |
| Interactive UI | `gradio` |
| PDF report | `fpdf2` |

## Quick start

### 1. Clone the repository
```bash
git clone git@github.com:DaniAngel79/voice-similarity-checker.git
cd voice-similarity-checker
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the notebook

Open `voice_similarity_checker.ipynb` in Jupyter and run all cells.

- **Demo cell:** runs automatically with the included sample WAV files
- **Last cell:** launches a Gradio UI at `http://localhost:7860` where you can upload any two audio files

## Sample files

`audio_path_1.wav` and `audio_path_2.wav` are short voice recordings included for testing purposes. Replace them with your own WAV or MP3 files to compare real voices.

## Output

- `spectrogram_output.png` — side-by-side log-frequency spectrograms of both audios
- `similarity_report.pdf` — full metrics report with verdict

## Similarity threshold

| MFCC Distance | Interpretation |
|---------------|----------------|
| < 0.15 | Same person |
| ≥ 0.15 | Different persons |

The threshold was calibrated empirically. Adjust it in `determine_similarity()` based on your use case.

## Project structure
```
voice-similarity-checker/
├── voice_similarity_checker.ipynb  # Main notebook
├── audio_path_1.wav                # Sample audio 1
├── audio_path_2.wav                # Sample audio 2
├── requirements.txt                # Python dependencies
├── .gitignore
└── README.md
```
