import torch
import torch.nn as nn
import torch.optim as optim
from model import AudioCNN
from dataset import ESC50Dataset
from torch.utils.data import DataLoader, random_split

# 1. Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AudioCNN(num_classes=50).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 2. Data Loaders (use your previous setup)
full_dataset = ESC50Dataset(csv_path='data/meta/esc50.csv', processed_dir='processed')

# 1. Define sizes
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size

# 2. Split using the variables
# Before splitting, set the seed
torch.manual_seed(42) 
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])


# 3. Double check (Good for debugging)
print(f"Train set size: {len(train_dataset)}")
print(f"Test set size: {len(test_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 3. Training Loop
print("Starting training...")
for epoch in range(10): # Run for 10 passes over the data
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()       # Reset gradients
        outputs = model(inputs)     # Forward pass
        loss = criterion(outputs, labels) # Calculate loss
        loss.backward()             # Backward pass
        optimizer.step()            # Update weights
        
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

# 4. Save the model
torch.save(model.state_dict(), "models/audio_model.pth")
print("Training complete! Model saved.")