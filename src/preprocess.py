import torchaudio
import torch

def get_spectrogram(filepath):
    waveform, sample_rate = torchaudio.load(filepath)
    # Convert to MelSpectrogram (AI "sees" this as an image)
    transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate)
    return transform(waveform)