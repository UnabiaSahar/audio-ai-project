import gradio as gr
import torch
import librosa
import numpy as np
import pandas as pd
from src.model import AudioCNN

# 1. Setup Environment Configurations
DEVICE = torch.device("cpu") 
MODEL_PATH = "models/audio_model.pth"
CSV_PATH = "data/meta/esc50.csv"

# Create explicit lookup mapping target integers to text categories
df = pd.read_csv(CSV_PATH)
target_to_category = dict(zip(df['target'].astype(int), df['category']))

# 2. Load the Weights into the Model Architecture securely
model = AudioCNN(num_classes=50)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval() 

# 3. Secure Inference Engine Pipeline
def predict_audio(audio_file):
    if audio_file is None:
        return {"Error": "No audio file provided"}

    # Replicate exactly the data-loading settings from your training stages
    y, sr = librosa.load(audio_file, sr=22050)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Standard Gaussian Normalization (Z-score scaling)
    mean = log_mel_spec.mean()
    std = log_mel_spec.std()
    normalized_spec = (log_mel_spec - mean) / (std + 1e-6)
    
    # Format dimensions: (Batch, Channel, Height, Width)
    tensor = torch.tensor(normalized_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    tensor = tensor.to(DEVICE)
    
    with torch.no_grad():
        output = model(tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        
    # Generate dictionary response for all 50 target boundaries
    confidences = {}
    for idx, prob in enumerate(probabilities):
        class_name = target_to_category[idx]
        confidences[class_name] = float(prob)
        
    return confidences

# 4. Construct Gradio Web Services with User Feedback/Flagging
demo = gr.Interface(
    fn=predict_audio, 
    inputs=gr.Audio(type="filepath"), 
    outputs=gr.Label(num_top_classes=3),
    title="ESC-50 Audio Classification System",
    description="Upload a 5-second audio clip. If it's an unknown or external sound, please use the feedback button below to mark if the top prediction was correct or incorrect.",
    
    # --- HUMAN FEEDBACK LOGGING CONFIGURATION ---
    flagging_mode="manual", # Enables the manual flag button underneath the outputs
    flagging_options=["True Prediction", "False Prediction"], # The choices the user can click
    flagging_dir="flagged_audio_dataset" # The local directory where data logs will be saved
)

if __name__ == "__main__":
    demo.launch()