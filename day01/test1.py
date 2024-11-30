import torch
import pandas as pd
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

torch.cuda.empty_cache()

# Assuming the annotations are in a text file
annotations_file = 'annotation.txt'
image_folder = './'

# Read the annotations
annotations = pd.read_csv(annotations_file, sep=' ', header=None)

# Define the transformations: ToTensor converts image to float32, and Normalize scales the values
data_transforms = transforms.Compose([
    transforms.ToTensor(),  # Converts PIL image to torch.FloatTensor and scales pixel values to [0, 1]
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Optional normalization
])

images = []
gaze_vectors = []

# Load images and gaze vectors
for idx, row in annotations.iterrows():
    # Format the image filename as '0001.jpg', '0002.jpg', ...
    image_filename = f'{idx + 1:04d}.jpg'  # +1 to start from 0001 instead of 0000
    image_path = os.path.join(image_folder, image_filename)
    
    # Open the image
    image = Image.open(image_path).convert('RGB')
    
    # Optionally apply transformations
    image = data_transforms(image)  # Uncomment if using transformations
    
    images.append(image)
    gaze_vectors.append(row[-3:].values)  # Gaze vector (last three values)


class GazeDataset(Dataset):
    def __init__(self, images, gaze_vectors):
        self.images = images
        self.gaze_vectors = gaze_vectors

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], torch.tensor(self.gaze_vectors[idx], dtype=torch.float32)

gaze_dataset = GazeDataset(images, gaze_vectors)

train_size = int(0.8 * len(gaze_dataset))
val_size = len(gaze_dataset) - train_size
train_dataset, val_dataset = random_split(gaze_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True)


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 56 * 56, 128)  # Adjust based on the output size of conv layers
        self.fc2 = nn.Linear(128, 3)  # Output layer (e.g., for 3 gaze vector components)
    
    def forward(self, x):
        # Forward through conv layers
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)  # Max pooling
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)  # Max pooling
        
        # Check the output shape of the conv layers
        print(f"Shape after conv layers: {x.shape}")
        
        # Flatten the output from conv layers
        x = x.view(x.size(0), -1)  # Flatten
        print(f"Shape after flattening: {x.shape}")
        
        # Forward through fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Instantiate the model
model = CNNModel()

# Create a dummy input with batch size 8 and image size 224x224
sample_input = torch.randn(8, 3, 224, 224)

# Forward pass
outputs = model(sample_input)


criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

num_epochs = 10

for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    
    # Training loop
    for images, gaze_vectors in train_loader:
        # Move data to GPU if using CUDA
        images = images.to(device)
        gaze_vectors = gaze_vectors.to(device)
        
        # Zero gradients from the previous step
        optimizer.zero_grad()
        
        # Forward pass: compute the output
        outputs = model(images)
        
        # Compute the loss
        loss = criterion(outputs, gaze_vectors)
        
        # Backward pass: compute gradients
        loss.backward()
        
        # Update the model parameters
        optimizer.step()
        
        # Accumulate the running loss
        running_loss += loss.item()
    
    # Calculate average loss over the epoch
    epoch_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

    # Optional: Validation loop (use a validation DataLoader)
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    with torch.no_grad():  # No need to compute gradients for validation
        for images, gaze_vectors in val_loader:
            images = images.to(device)
            gaze_vectors = gaze_vectors.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, gaze_vectors)
            val_loss += loss.item()
    
    val_loss = val_loss / len(val_loader)
    print(f'Validation Loss: {val_loss:.4f}')
