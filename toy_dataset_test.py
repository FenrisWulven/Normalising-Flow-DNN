import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from typing import List, Tuple, Callable

# Generate synthetic 2D data like one moon
np.random.seed(0)
n_samples = 500
a = -0.5
b = -0.2
c = 4
x = np.linspace(-5, 5, n_samples)
y = a * x**2 + b * x + c
noise_scale = 1.0
y_noisy = y + noise_scale * np.random.randn(len(x))
data = np.vstack((x, y_noisy)).T

# Plot the toy dataset
plt.scatter(data[:, 0], data[:, 1], s=10)
plt.title("Toy Dataset with Negative 2nd-Degree Polynomial")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()

# Preprocess the data (standardize)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data = scaler.fit_transform(data)

# Convert data to PyTorch tensor
X_tensor = torch.FloatTensor(data)

# Define the Normalizing Flow model template
class NormalizingFlow(nn.Module):

    def __init__(self, flows: List[nn.Module]):
        super(NormalizingFlow, self).__init__()
        self.flows = nn.ModuleList(flows)

    def f(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z, sum_log_abs_det = x, torch.zeros(x.size(0)).to(x.device)
        for flow in self.flows:
            z, log_abs_det = flow.f(z)
            sum_log_abs_det += log_abs_det

        return z, sum_log_abs_det

    def g(self, z: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = z
            for flow in reversed(self.flows):
                x = flow.g(x)

        return x

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        z, log_abs_det = self.f(x)
        return log_abs_det

# Define the AffineCouplingLayer
class AffineCouplingLayer(nn.Module):

    def __init__(
            self,
            theta: nn.Module,
            split: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]
    ):
        super(AffineCouplingLayer, self).__init__()
        self.theta = theta
        self.split = split

    def f(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x2, x1 = self.split(x)
        t, s = self.theta(x1)
        z1, z2 = x1, x2 * torch.exp(s) + t
        log_det = s.sum(-1)
        return torch.cat((z1, z2), dim=-1), log_det

    def g(self, z: torch.Tensor) -> torch.Tensor:
        z1, z2 = self.split(z)
        t, s = self.theta(z1)
        x1, x2 = z1, (z2 - t) * torch.exp(-s)
        return torch.cat((x2, x1), dim=-1)

# Define the Conditioner
class Conditioner(nn.Module):

    def __init__(
            self, in_dim: int, out_dim: int,
            num_hidden: int, hidden_dim: int,
            num_params: int
    ):
        super(Conditioner, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_dim, hidden_dim))
        for _ in range(num_hidden):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, out_dim * num_params))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = F.leaky_relu(layer(x))
        batch_params = self.layers[-1](x).reshape(x.size(0), -1, out_dim)
        params = batch_params.chunk(self.num_params, dim=-1)
        return [p.squeeze(-1) for p in params]

# Create the flows
out_dim = 2
num_params = 2
hidden_dim = 32
num_hidden = 2

conditioner = Conditioner(in_dim=2, out_dim=out_dim, num_hidden=num_hidden, hidden_dim=hidden_dim, num_params=num_params)
affine_coupling = AffineCouplingLayer(conditioner, split=lambda x: x.chunk(2, dim=-1))
flows = [affine_coupling for _ in range(5)]

# Create the Normalizing Flow model
flow_model = NormalizingFlow(flows)

# Train the model (for demonstration purposes, you can adjust the training process)
optimizer = torch.optim.Adam(flow_model.parameters(), lr=0.001)

losses = []
num_epochs = 1000

for epoch in range(num_epochs):
    optimizer.zero_grad()
    log_prob = flow_model.log_prob(X_tensor)
    loss = -torch.mean(log_prob)
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if epoch % 100 == 0:
        print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item()}")

# Generate samples from the trained model
with torch.no_grad():
    num_samples = 1000
    samples = flow_model.sample(num_samples=num_samples)
    samples = samples.numpy()

# Plot the loss
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()

# Plot the original data and generated samples
plt.scatter(X_tensor[:, 0], X_tensor[:, 1], s=10, label="Original Data")
plt.scatter(samples[:, 0], samples[:, 1], s=10, label="Generated Samples")
plt.title("Original Data vs. Generated Samples")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()
plt.show()
