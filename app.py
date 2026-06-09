import gradio as gr
import torch
import librosa
import numpy as np
import pandas as pd
from src.model import AudioCNN

# 1. Setup
DEVICE = torch.device("cpu") # Hugging Face spaces usually run on CPU
MODEL_PATH = "models/audio_model.pth"
CSV_PATH = "data/meta/esc50.csv"

# Load labels from the CSV
df = pd.read_csv(CSV_PATH)
labels = sorted(df['category'].unique())

# 2. Load the Model
model = AudioCNN(num_classes=50)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# 3. Inference Function
def predict_audio(audio_file):
    # Load and Preprocess exactly like in training
    y, sr = librosa.load(audio_file, sr=22050)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Add batch and channel dimensions: (1, 1, 128, 216)
    # Note: If your audio isn't exactly 5s, you might need padding here!
    tensor = torch.tensor(log_mel_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    with torch.no_grad():
        output = model(tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
    # Get top prediction
    top_prob, top_idx = torch.max(probabilities, 0)
    return {labels[top_idx]: float(top_prob)}

# 4. Launch Interface
demo = gr.Interface(
    fn=predict_audio, 
    inputs=gr.Audio(type="filepath"), 
    outputs=gr.Label(num_top_classes=3)
)

if __name__ == "__main__":
    demo.launch()