# audio-ai-project

Environmental sound classification on the [ESC-50](https://github.com/karolpiczak/ESC-50) dataset using a custom convolutional neural network (CNN) built with PyTorch. The project covers the full machine-learning pipeline — preprocessing, dataset loading, training, evaluation, inference — and ships a Gradio web interface for interactive prediction and human-in-the-loop correction data collection.

## Features

- **Custom CNN (`AudioCNN`)** — 4 convolutional blocks (BatchNorm + ReLU + pooling) with adaptive pooling and a fully-connected classifier head.
- **Mel-spectrogram preprocessing** — audio is loaded at 22.05 kHz, converted to log Mel-spectrograms (128 Mel bands), and Z-score normalized before being saved as `.npy` features.
- **50-class classification** — maps audio to one of the 50 ESC-50 categories (dog bark, rain, helicopter, etc.).
- **Training & evaluation** — 80/20 train/test split, Adam optimizer, CosineAnnealingLR scheduler, and test accuracy reporting.
- **CLI inference** — predict the class and confidence of a single audio file from the terminal.
- **Gradio web app** — upload audio, get live predictions, and flag misclassifications to build a corrected dataset (`external_esc50.csv`) that can be used for further training.

## Project Structure

```
audio-ai-project/
├── app.py               # Gradio web interface + custom flagging callback (data collection)
├── requirements.txt     # Python dependencies
├── test_loader.py       # Smoke test for the dataset loader
├── data/
│   ├── audio/           # Raw ESC-50 audio files (.wav) — expected location
│   ├── external_audio/  # Archived audio flagged as misclassified via the app
│   └── meta/
│       ├── esc50.csv           # ESC-50 labels/metadata
│       └── external_esc50.csv  # New ground-truth labels collected via the app
├── processed/           # Precomputed .npy Mel-spectrogram features
├── models/
│   └── audio_model.pth  # Trained model weights
└── src/
    ├── preprocess.py    # Mel-spectrogram extraction + normalization -> .npy
    ├── dataset.py       # PyTorch Dataset for loading .npy features
    ├── model.py         # AudioCNN definition
    ├── train.py         # Training loop
    ├── eval.py          # Accuracy evaluation on the held-out test split
    └── inference.py     # Single-file CLI prediction
```

## Installation

```bash
# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # Linux / macOS

# Install dependencies
pip install -r requirements.txt
```

## Dataset Setup

The project expects the ESC-50 dataset (2000 audio clips across 50 classes, 40 clips per class) with its `esc50.csv` metadata.

1. Download the dataset from the [official ESC-50 repository](https://github.com/karolpiczak/ESC-50).
2. Place the audio files in `data/audio/` and the metadata at `data/meta/esc50.csv`.

## Usage

### 1. Preprocess

Convert raw audio into normalized Mel-spectrogram features:

```bash
python src/preprocess.py
```

This reads every `.wav` in `data/audio/` and writes one `.npy` feature map per file to `processed/`.

### 2. Verify the data loader

```bash
python test_loader.py
```

### 3. Train

```bash
python src/train.py
```

Trains `AudioCNN` for 50 epochs and saves the weights to `models/audio_model.pth`.

### 4. Evaluate

```bash
python src/eval.py
```

Reports accuracy on the held-out 20% test split.

### 5. Single-file inference (CLI)

```bash
python src/inference.py
```

Edit the `sample_file` path inside the script to test your own audio.

### 6. Web interface (Gradio)

```bash
python app.py
```

Open the printed local URL. Upload an audio file to see the model's prediction, then flag it as **True Prediction** or **False Prediction**. When you flag a misclassification, use the dropdown to supply the correct label — the app archives the audio to `data/external_audio/`, preprocesses it into `processed/`, and appends a new ground-truth row to `data/meta/external_esc50.csv` (useful for active-learning / retraining loops).

## Pipeline Details

| Stage | Description |
| --- | --- |
| Sampling rate | 22.05 kHz |
| Feature | Log Mel-spectrogram, 128 Mel bands |
| Normalization | Z-score (`(x - mean) / (std + 1e-6)`) |
| Model | `AudioCNN` — 4 conv blocks → adaptive pooling → 2 FC layers (dropout 0.5) |
| Optimizer | Adam (lr 0.001) + CosineAnnealingLR |
| Loss | CrossEntropyLoss |
| Classes | 50 (ESC-50 targets) |

## License

[MIT](LICENSE) © 2026 Unabia Sahar
