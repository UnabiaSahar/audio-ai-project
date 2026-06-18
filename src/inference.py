import torch
import librosa
import numpy as np
import pandas as pd
from model import AudioCNN

def predict_audio(audio_file_path, model_path, csv_path):
    # 1. Setup device and load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioCNN(num_classes=50).to(device)
    
    # Load weights and set to evaluation mode
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 2. Replicate the Training Preprocessing Pipeline
    # Load audio at the exact training sample rate
    y, sr = librosa.load(audio_file_path, sr=22050) 
    
    # Extract Mel-Spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    
    # Convert to Log-Scale (dB)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Standard Gaussian Normalization (Z-score) matching preprocess.py
    mean = log_mel_spec.mean()
    std = log_mel_spec.std()
    normalized_spec = (log_mel_spec - mean) / (std + 1e-6)
    
    # 3. Format tensor for PyTorch CNN (Batch, Channel, Height, Width)
    input_tensor = torch.tensor(normalized_spec, dtype=torch.float32)
    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0) # Adds Batch and Channel dimensions
    input_tensor = input_tensor.to(device)
    
    # 4. Model Prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        # Apply softmax to get probabilities (optional, but good for confidence scores)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
        
    # 5. Map index back to the actual string class name
    # Using the ESC-50 csv metadata to get the human-readable label
    df = pd.read_csv(csv_path)
    
    # Create a clean dictionary mapping target integer to category string
    target_to_category = dict(zip(df['target'], df['category']))
    predicted_label = target_to_category[predicted_idx.item()]
    
    print(f"--- Inference Result ---")
    print(f"File: {audio_file_path}")
    print(f"Predicted Class: {predicted_label}")
    print(f"Confidence Score: {confidence.item() * 100:.2f}%")
    
    return predicted_label, confidence.item()

if __name__ == "__main__":
    # Example usage: Change this path to test an audio file
    sample_file = "data/audio/5-263775-B-26.wav" 
    
    predict_audio(
        audio_file_path=sample_file,
        model_path="models/audio_model.pth",
        csv_path="data/meta/esc50.csv"
    )