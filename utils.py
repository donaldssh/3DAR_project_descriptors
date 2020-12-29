import torch
from torch import nn
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class EncoderConv(nn.Module):
    
    def __init__(self, encoded_space_dim, conv1_ch, conv2_ch, conv3_ch, fc_ch):
        super().__init__()
        
        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            # First convolutional layer
            nn.Conv2d(4, conv1_ch, 3, stride=1, padding=1),
            nn.ReLU(True),
            # Second convolutional layer
            nn.Conv2d(conv1_ch, conv2_ch, 3, stride=1, padding=1),
            nn.ReLU(True),
            # Third convolutional layer
            nn.Conv2d(conv2_ch, conv3_ch, 3, stride=1, padding=0),  #make it 3x3
            nn.ReLU(True)
        )
        
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)

        ### Linear section
        self.encoder_lin = nn.Sequential(
            # First linear layer
            nn.Linear(2 * 2 * conv3_ch, fc_ch),
            nn.ReLU(True),
            # Second linear layer
            nn.Linear(fc_ch, encoded_space_dim)
        )
        
    def forward(self, x):
        # Apply convolutions
        x = self.encoder_cnn(x)
        # Flatten
        # print(np.shape(x))
        x = self.flatten(x)
        # # Apply linear layers
        x = self.encoder_lin(x)
        return x


class DecoderConv(nn.Module):
    
    def __init__(self, encoded_space_dim, conv1_ch, conv2_ch, conv3_ch, fc_ch):
        super().__init__()

        ### Linear section
        self.decoder_lin = nn.Sequential(
            # First linear layer
            nn.Linear(encoded_space_dim, fc_ch),
            nn.ReLU(True),
            # Second linear layer
            nn.Linear(fc_ch, 2 * 2 * conv3_ch),
            nn.ReLU(True)
        )

        ### Unflatten
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(conv3_ch, 2, 2))

        ### Convolutional section
        self.decoder_conv = nn.Sequential(
            # First transposed convolution
            nn.ConvTranspose2d(conv3_ch, conv2_ch, 3, stride=1, output_padding=0),
            nn.ReLU(True),
            # Second transposed convolution
            nn.ConvTranspose2d(conv2_ch, conv1_ch, 3, stride=1, padding=1, output_padding=0),
            nn.ReLU(True),
            # Third transposed convolution
            nn.ConvTranspose2d(conv1_ch, 4, 3, stride=1, padding=1, output_padding=0)
        )
        
    def forward(self, x):
        # Apply linear layers
        x = self.decoder_lin(x)
        # Unflatten
        x = self.unflatten(x)
        # print(np.shape(x))
        # Apply transposed convolutions
        x = self.decoder_conv(x)
        # # Apply a sigmoid to force the output to be between 0 and 1 
        # x = torch.sigmoid(x)
        return x

class EncoderLin(nn.Module):
    
    def __init__(self, encoded_space_dim, fc1_ch, fc2_ch, fc3_ch, fc4_ch):
        super().__init__()

        ### Linear section
        self.encoder_lin = nn.Sequential(
            # First linear layer
            nn.Linear(64, fc1_ch),
            nn.ReLU(True),
            # nn.BatchNorm1d(fc1_ch)
            # Second linear layer
            nn.Linear(fc1_ch, fc2_ch),
            nn.ReLU(True),
            # Third linear level
            nn.Linear(fc2_ch, fc3_ch),
            nn.ReLU(True),
            nn.Linear(fc3_ch, fc4_ch),
            nn.ReLU(True),
            nn.Linear(fc4_ch, encoded_space_dim)
        )
        
    def forward(self, x):
        # Apply linear layers
        x = self.encoder_lin(x)
        return x


class DecoderLin(nn.Module):
    
    def __init__(self, encoded_space_dim, fc1_ch, fc2_ch, fc3_ch, fc4_ch):
        super().__init__()

        ### Linear section
        self.decoder_lin = nn.Sequential(
            # First linear layer
            nn.Linear(encoded_space_dim, fc4_ch),
            nn.ReLU(True),
            nn.Linear(fc4_ch, fc3_ch),
            nn.ReLU(True),
            # Second linear layer
            nn.Linear(fc3_ch, fc2_ch),
            nn.ReLU(True),
            nn.Linear(fc2_ch, fc1_ch),
            nn.ReLU(True),
            nn.Linear(fc1_ch, 64)
        )
        
    def forward(self, x):
        # Apply linear layers
        x = self.decoder_lin(x)
        return x

class SurfDataset(Dataset):

    def __init__(self, data, transform=None):
        self.transform = transform
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #return a nparray from the dataframe
        sample = self.data.iloc[idx,:].to_numpy()
        if self.transform:
            sample = self.transform(sample)
        return sample

class NpToTensor():
    def __call__(self, array):
        return torch.from_numpy(array)
    
class Surf3DReshape():
    def __call__(self, desc):
        channels = []
        for i in range(4):
            channels.append(np.reshape(desc[i:64:4], (4,4)))
        channels = np.dstack(channels)
        channels = np.transpose(channels, (2,0,1))
        return channels

class Surf3DInverseReshape():
    def __call__(self, desc):
        channel = []
        for i in range(16):
            for j in range(4):
                channel.append(desc[j*16+i])
                
        return np.array(channel)
