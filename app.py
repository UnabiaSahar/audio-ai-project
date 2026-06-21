import os
import csv
import uuid
import torch
import librosa
import soundfile as sf
import numpy as np
import pandas as pd
import gradio as gr
from src.model import AudioCNN

# 1. Setup Environment Configurations
DEVICE = torch.device("cpu") 
MODEL_PATH = "models/audio_model.pth"
CSV_PATH = "data/meta/esc50.csv"
EXTERNAL_CSV_PATH = "data/meta/external_esc50.csv"
PROCESSED_DIR = "processed"

# Create structural mapping setups
df = pd.read_csv(CSV_PATH)
target_to_category = dict(zip(df['target'].astype(int), df['category']))
all_categories = sorted(df['category'].unique())

# 2. Senior Implementation: Custom ESC-50 Dataset Logger Class
class ESC50CustomLogger(gr.FlaggingCallback):
    def __init__(self, external_csv, processed_path, target_map):
        self.external_csv = external_csv
        self.processed_path = processed_path
        # Reverse lookup: text category string -> integer target index
        self.category_to_target = {v: k for k, v in target_map.items()}

    def setup(self, components, flagging_dir):
        self.components = components
        os.makedirs(os.path.dirname(self.external_csv), exist_ok=True)
        os.makedirs(self.processed_path, exist_ok=True)
        
        # Instantiate structural headers if file doesn't exist yet
        if not os.path.exists(self.external_csv):
            with open(self.external_csv, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['filename', 'fold', 'target', 'category', 'esc10', 'src_file', 'take'])

    def flag(self, flag_data, flag_option, username=None):
        # Unpack securely. Depending on Gradio version variations, 
        # flag_data contains: [audio_input, dropdown_input, model_output]
        audio_input = flag_data[0]
        dropdown_value = flag_data[1]
        model_output = flag_data[2] 

        if not audio_input:
            print("[ERROR] No audio input tracked.")
            return

        # 1. Handle Gradio's new FileData dict structure safely
        if isinstance(audio_input, dict) and 'path' in audio_input:
            audio_path = audio_input['path']
        else:
            audio_path = str(audio_input)

        if not os.path.exists(audio_path):
            print(f"[ERROR] Target file does not exist at path: {audio_path}")
            return

        # 2. Extract top prediction label from Gradio's Label layout dictionary
        if isinstance(model_output, dict) and "label" in model_output:
            top_model_prediction = model_output["label"]
        elif isinstance(model_output, dict):
            top_model_prediction = max(model_output, key=lambda k: model_output[k] if isinstance(model_output[k], (int, float)) else 0)
        else:
            top_model_prediction = str(model_output)

        # 3. FIX THE PROBLEM: Ensure the corrected label is explicitly prioritized on False Prediction
        if flag_option == "True Prediction":
            category = top_model_prediction
        else:  # "False Prediction"
            # If the user forgot to change the dropdown, warn them, but fall back gracefully
            if dropdown_value == "N/A - Prediction is True" or not dropdown_value:
                print("[WARNING] User flagged as False Prediction but did not change the correction dropdown!")
                category = top_model_prediction  # fallback
            else:
                # SUCCESS: Capture the correct label selected by the user
                category = dropdown_value

        # Look up target integer matching the finalized category string
        target_idx = self.category_to_target.get(category, 0)

        # 4. Enforce unified ESC-50 formatting rules
        unique_suffix = str(uuid.uuid4())[:8]
        standardized_filename = f"external_{unique_suffix}.wav"
        
        archive_dir = os.path.join("data", "external_audio")
        os.makedirs(archive_dir, exist_ok=True)
        dest_wav_path = os.path.join(archive_dir, standardized_filename)

        try:
            # Resample and convert whatever container (.mp3, .aac, .wav) safely to 22050Hz
            y, sr = librosa.load(audio_path, sr=22050)
            sf.write(dest_wav_path, y, sr, format='WAV')

            # Preprocess features instantly so they match your training specs
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            normalized_spec = (log_mel_spec - log_mel_spec.mean()) / (log_mel_spec.std() + 1e-6)
            
            # Save the feature map straight to training target
            npy_filename = standardized_filename.replace('.wav', '.npy')
            np.save(os.path.join(self.processed_path, npy_filename), normalized_spec)

            # Append metadata to match original structural rows
            with open(self.external_csv, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([standardized_filename, 6, target_idx, category, False, 0, 'X'])
                
            print(f"[SUCCESS] Ground truth logged and preprocessed: {standardized_filename} -> target: {target_idx} ({category})")
        except Exception as e:
            print(f"[ERROR] Failed tracking operations: {e}")
                 
# 3. Load Neural Network Setup
model = AudioCNN(num_classes=50)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval() 

def predict_audio(audio_file, correct_label_dropdown):
    if audio_file is None: return {"Error": "No audio file"}
    y, sr = librosa.load(audio_file, sr=22050)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    normalized_spec = (log_mel_spec - log_mel_spec.mean()) / (log_mel_spec.std() + 1e-6)
    
    tensor = torch.tensor(normalized_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        
    return {target_to_category[idx]: float(prob) for idx, prob in enumerate(probabilities)}

# 4. Bind Everything to Interface Definition
audio_input = gr.Audio(type="filepath", label="Upload Audio")
dropdown_input = gr.Dropdown(
    choices=["N/A - Prediction is True"] + all_categories, 
    value="N/A - Prediction is True", 
    label="Correct Label (Only fill this out if you are marking a 'False Prediction')"
)

custom_logger = ESC50CustomLogger(EXTERNAL_CSV_PATH, PROCESSED_DIR, target_to_category)

demo = gr.Interface(
    fn=predict_audio, 
    inputs=[audio_input, dropdown_input], 
    outputs=gr.Label(num_top_classes=1),
    title="ESC-50 Audio Classification System",
    description="Mark your prediction as True or False below. If False, make sure to specify the correction label from the dropdown list.",
    flagging_mode="manual", 
    flagging_options=["True Prediction", "False Prediction"],
    flagging_callback=custom_logger # Binds the custom logging engine
)

if __name__ == "__main__":
    demo.launch()