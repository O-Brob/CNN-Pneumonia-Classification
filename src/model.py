# ===== Imports ===== #
import torch
import torch.nn as nn
import torch.nn.functional as F

from src import config

# ===== Convolutional Neural Network Class ===== #
class CNN(nn.Module):
    # Define Convolutional Neural Network structure (VGG-inspired)
    def __init__(self):
        super(CNN, self).__init__()
        
        # Convolutional Block 1:
        self.conv1  = nn.Conv2d(1, 32, kernel_size=3, padding="same")
        self.batch1 = nn.BatchNorm2d(32)
        
        # Convolutional Block 2:
        self.conv2  = nn.Conv2d(32, 64, kernel_size=3, padding="same")
        self.batch2 = nn.BatchNorm2d(64)
        
        #Convolutional Block 3:
        self.conv3  = nn.Conv2d(64, 128, kernel_size=3, padding="same")
        self.batch3 = nn.BatchNorm2d(128)
        
        # Pooling and dropout to use in forward propagation
        self.pool    = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(config.DROPOUT)
        
        # Classification Head:
        # Image rescale size: 256 --> 128 -> 64 -> 32 after the Conv layers.
        self.fullcon1 = nn.Linear(128 * 32 * 32, 512)
        self.fullcon2 = nn.Linear(512, 1) # binary classification
    
    # Forward propagation for the model
    # Each block goes through ConvN -> BN -> ReLU -> Max Pooling
    def forward(self, x):
        # Forward propagate block 1
        x = self.conv1(x)
        x = self.batch1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        
        # Forward propagate block 2
        x = self.conv2(x)
        x = self.batch2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        
        # Forward propagate block 3
        x = self.conv3(x)
        x = self.batch3(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        
        # Flattening; (128, 32, 32) -> (131072)
        x = torch.flatten(x, 1)
        
        # Fully connected layers
        x = self.fullcon1(x)
        x = F.relu(x)
        x = self.fullcon2(x)
        
        return x
