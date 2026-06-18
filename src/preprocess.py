import librosa
import numpy as np
import os
from pathlib import Path

def extract_and_normalize_mel_spectrogram(file_path):
    # 1. Load the audio file (ESC-50 clips are exactly 5 seconds)
    y, sr = librosa.load(file_path, sr=22050) 
    
    # 2. Extract Mel-Spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    
    # 3. Convert to Log-Scale (dB)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    # 4. Standard Gaussian Normalization (Z-score)
    # This centers the data around 0 with a standard deviation of 1
    mean = log_mel_spec.mean()
    std = log_mel_spec.std()
    
    # 1e-6 (epsilon) protects against dividing by zero on completely silent frames
    normalized_spec = (log_mel_spec - mean) / (std + 1e-6)
    
    return normalized_spec

# Create destination folder if it doesn't exist
output_dir = Path("processed")
output_dir.mkdir(exist_ok=True)

# Loop through your dataset
data_path = Path("data/audio/")
print("Starting preprocessing and normalization...")

for file in data_path.glob("*.wav"):
    features = extract_and_normalize_mel_spectrogram(file)
    
    # Save as .npy (Numpy format) for fast loading during training
    np.save(output_dir / f"{file.stem}.npy", features)

print("Preprocessing complete! All files normalized and saved.")