import torch
from model import AudioCNN
from dataset import ESC50Dataset
from torch.utils.data import DataLoader, random_split

def evaluate():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioCNN(num_classes=50).to(device)
    model.load_state_dict(torch.load("models/audio_model.pth"))
    model.eval() # Set to evaluation mode
    
    # Load Test Data
    full_dataset = ESC50Dataset(csv_path='data/meta/esc50.csv', processed_dir='processed')
    _, test_dataset = random_split(full_dataset, [int(0.8 * len(full_dataset)), len(full_dataset) - int(0.8 * len(full_dataset))])
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Evaluate
    correct = 0
    total = 0
    with torch.no_grad(): # Disable gradient calculation for speed
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    print(f'Accuracy on test set: {100 * correct / total:.2f}%')

if __name__ == "__main__":
    evaluate()