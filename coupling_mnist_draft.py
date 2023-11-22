import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import os
from torch.utils.data.dataset import Subset

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
num_epochs = 20
batch_size = 64
latent_dim = 128  # Adjust as needed
num_classes = 10
learning_rate = 0.001
num_ensemble_models = 5  # Number of models in the ensemble

# CNN Model
class CNN(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        # Define the CNN architecture here
        # ...

    def forward(self, x):
        # Forward pass through the CNN
        # ...
        return x

# Simple Classifier Model
class SimpleClassifier(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super(SimpleClassifier, self).__init__()
        self.cnn = CNN(latent_dim)
        self.fc = nn.Linear(latent_dim, num_classes)

    def forward(self, x):
        features = self.cnn(x)
        output = self.fc(features)
        return F.softmax(output, dim=1)

# Instantiate Ensemble Models
ensemble_models = [SimpleClassifier(latent_dim, num_classes).to(device) for _ in range(num_ensemble_models)]
optimizers = [torch.optim.Adam(model.parameters(), lr=learning_rate) for model in ensemble_models]

# Loss Function
criterion = nn.CrossEntropyLoss()

# Data loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Training Function
def train_model(model, optimizer, train_loader, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            # Print loss, accuracy, etc. as needed

# Train each model in the ensemble
for model, optimizer in zip(ensemble_models, optimizers):
    train_model(model, optimizer, train_loader, num_epochs)

# Evaluation Function
def evaluate_ensemble(ensemble_models, test_loader):
    for model in ensemble_models:
        model.eval()
    
    total_correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            ensemble_outputs = torch.stack([model(data) for model in ensemble_models])
            ensemble_output = torch.mean(ensemble_outputs, dim=0)
            pred = ensemble_output.argmax(dim=1, keepdim=True)
            total_correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = total_correct / len(test_loader.dataset)
    return accuracy

# Evaluate Ensemble
ensemble_accuracy = evaluate_ensemble(ensemble_models, test_loader)
print(f'Ensemble Accuracy: {ensemble_accuracy:.4f}')
