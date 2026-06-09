from src.dataset import ESC50Dataset
import os

# Define paths
# Ensure these paths match where you actually have your data
CSV_PATH = 'data/meta/esc50.csv' 
PROCESSED_DIR = 'processed'

def test_loading():
    # 1. Initialize the Dataset
    try:
        dataset = ESC50Dataset(csv_path=CSV_PATH, processed_dir=PROCESSED_DIR)
        print(f"Successfully initialized dataset with {len(dataset)} samples.")
        
        # 2. Get the first sample
        sample, label = dataset[0]
        
        # 3. Print the shape
        print(f"Sample shape: {sample.shape}")
        print(f"Label index: {label}")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find files. Check your paths! Details: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    test_loading()