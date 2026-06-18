import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
   
# In dataset.py 
class ESC50Dataset(Dataset):

    def __init__(self, csv_path, processed_dir):
        self.df = pd.read_csv(csv_path)
        self.processed_dir = processed_dir
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row['filename'].replace('.wav', '.npy')
        
        # Correct: Use the pre-existing target mapping 0-49
        label = int(row['target']) 
        
        spec = np.load(os.path.join(self.processed_dir, filename))
        
        # Apply Z-score normalization inline or during preprocessing
        spec = (spec - spec.mean()) / (spec.std() + 1e-6)
        
        spec = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)
        return spec, label