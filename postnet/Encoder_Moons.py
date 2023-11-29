import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder_Moons(nn.Module):

    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.fc1 = nn.Linear(2, 32)
        self.act1 = nn.ReLU()
        #self.act1 = nn.LeakyReLU()
        #self.act1 = nn.Sigmoid()
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(32,16)
        self.act2 = nn.ReLU()
        #self.act2 = nn.LeakyReLU()
        #self.act2 = nn.Sigmoid()
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(16, latent_dim)
        self.batchnorm = nn.BatchNorm1d(latent_dim) #num_classes
        # increase hidden layers and neurons for more complex data patterns
        # use different activation functions that are non-linear such as tanh, sigmoid, leaky relu, elu

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.batchnorm(x)
        return x