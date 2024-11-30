import torch
import pandas as pd
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

# Clear GPU memory cache
torch.cuda.empty_cache()

# Assuming the annotations are in a text file
annotations_file = 'annotation.txt'  # Replace with your actual annotation file path
image_folder = './'  # Replace with your actual image folder path

# Read the annotations
annotations = pd.read_csv(annotations_file, sep=' ', header=None)

# Define the transformations: ToTensor converts image to float32, and Normalize scales the values
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),  # Converts PIL image to torch.FloatTensor and scales pixel values to [0, 1]
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Optional normalization
])

# Load images and gaze vectors
images = []
gaze_vectors = []
for idx, row in annotations.iterrows():
    # Format the image filename as '0001.jpg', '0002.jpg', ...
    image_filename = f'{idx + 1:04d}.jpg'
    image_path = os.path.join(image_folder, image_filename)

    # Open the image
    image = Image.open(image_path).convert('RGB')

    # Apply transformations
    image = data_transforms(image)

    images.append(image)
    gaze_vectors.append(row[-3:].values)  # Gaze vector (last three values)

# Dataset definition
class GazeDataset(Dataset):
    def __init__(self, images, gaze_vectors):
        self.images = images
        self.gaze_vectors = gaze_vectors

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], torch.tensor(self.gaze_vectors[idx], dtype=torch.float32)

gaze_dataset = GazeDataset(images, gaze_vectors)

# Split into training and validation datasets
train_size = int(0.8 * len(gaze_dataset))
val_size = len(gaze_dataset) - train_size
train_dataset, val_dataset = random_split(gaze_dataset, [train_size, val_size])

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)  # Adjust batch size if needed
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True)

# CNN model definition
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Dynamically calculate the input size for fully connected layers
        self.fc1 = None
        self.fc2 = nn.Linear(128, 3)  # Output layer for 3 gaze vector components

        # Initialize the fully connected layer
        self._initialize_fc1()

    def _initialize_fc1(self):
        # Create a dummy tensor with the size of the input images
        dummy_input = torch.randn(1, 3, 224, 224)  # Adjust the size as needed
        # Pass it through the convolutional layers
        x = F.relu(self.conv1(dummy_input))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        # Calculate the flattened size after the conv layers
        flatten_size = x.numel()  # Get the number of elements
        self.fc1 = nn.Linear(flatten_size, 128)  # Set the input size for fc1

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Instantiate model, loss function, and optimizer
model = CNNModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Training parameters
num_epochs = 10

# Training and validation loop
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0

    # Training loop
    for images, gaze_vectors in train_loader:
        # Move data to GPU if available
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
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')

    # Validation loop
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    with torch.no_grad():  # No need to compute gradients during validation
        for images, gaze_vectors in val_loader:
            images = images.to(device)
            gaze_vectors = gaze_vectors.to(device)

            outputs = model(images)
            loss = criterion(outputs, gaze_vectors)
            val_loss += loss.item()

    # Calculate average validation loss
    val_loss = val_loss / len(val_loader)
    print(f'Validation Loss: {val_loss:.4f}')
