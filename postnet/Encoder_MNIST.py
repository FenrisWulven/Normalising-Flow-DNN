import torch
import torch.nn as nn

class Encoder_MNIST(nn.Module):

    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) 
        )

        self.linear_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, latent_dim),
            nn.BatchNorm1d(latent_dim)
        )

    def forward(self, x):
        x = self.conv_block(x)
        #print("Shape after conv_block:", x.shape) #64batchsize x 64channels x 1height x 1width
        #x = x.view(x.size(0), -1) # flatten
        x = self.linear_block(x)
        return x