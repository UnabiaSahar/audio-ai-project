import librosa
import numpy as np
import os
from pathlib import Path

def extract_mel_spectrogram(file_path):
    # 1. Load the audio file
    y, sr = librosa.load(file_path, sr=22050) 
    
    # 2. Extract Mel-Spectrogram
    # n_mels=128 is a standard height for the image-like output
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    
    # 3. Convert to Log-Scale (dB)
    # Human hearing is logarithmic; this makes the features more useful for the AI
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    return log_mel_spec

# Loop through your dataset
data_path = Path("data/audio/")
for file in data_path.glob("*.wav"):
    features = extract_mel_spectrogram(file)
    # Save as .npy (Numpy format) for fast loading during training
    np.save(f"processed/{file.stem}.npy", features)