import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os

class ESC50Dataset(Dataset):
    def __init__(self, csv_path, processed_dir):
        self.df = pd.read_csv(csv_path)
        self.processed_dir = processed_dir
        
        # Create a mapping from category name to integer
        self.labels = self.df['category'].unique()
        self.label_to_idx = {label: i for i, label in enumerate(self.labels)}
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Get filename and label from row
        row = self.df.iloc[idx]
        filename = row['filename'].replace('.wav', '.npy')
        label = self.label_to_idx[row['category']]
        
        # Load the pre-saved spectrogram
        spec = np.load(os.path.join(self.processed_dir, filename))
        
        # CNNs in PyTorch expect a channel dimension (Batch, Channel, Height, Width)
        # We add the channel dimension: (1, 128, X)
        spec = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)
        
        return spec, label