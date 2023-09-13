import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic 2D data like one moon
np.random.seed(0)
n_samples = 500
# Define the coefficients of the negative 2nd-degree polynomial equation
a = -0.3
b = -0.5
c = 5
# Generate x-values
x = np.linspace(-5, 5, 100)
# Calculate y-values using the polynomial equation
y = a * x**2 + b * x + c
# Add some random noise to the y-values
noise_scale = 1.0
y_noisy = y + noise_scale * np.random.randn(len(x))
# Create the toy dataset
data = np.vstack((x, y_noisy)).T
# Plot the toy dataset
# plt.scatter(data[:, 0], data[:, 1], s=10)
# plt.title("Toy Dataset with Negative 2nd-Degree Polynomial")
# plt.xlabel("X-axis")
# plt.ylabel("Y-axis")
# plt.show()

#%%

# Define a simple affine coupling layer
class AffineCoupling(nn.Module):
    def __init__(self, in_features, hidden_dim=16):
        super(AffineCoupling, self).__init__()
        self.scale_net = nn.Sequential(
            nn.Linear(in_features // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_features // 2),
            nn.Tanh()
        )
        self.translation_net = nn.Sequential(
            nn.Linear(in_features // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_features // 2)
        )

    def forward(self, x, reverse=False):
        x1, x2 = x.chunk(2, dim=1)
        if not reverse:
            scale = self.scale_net(x1)
            translation = self.translation_net(x1)
            y1 = x1
            y2 = x2 * scale.exp() + translation
        else:
            scale = self.scale_net(x1)
            translation = self.translation_net(x1)
            y1 = x1
            y2 = (x2 - translation) / scale.exp()
        return torch.cat([y1, y2], dim=1)

# Define a normalizing flow model
class NormalizingFlow(nn.Module):
    def __init__(self, num_coupling_layers, in_features):
        super(NormalizingFlow, self).__init__()
        self.num_coupling_layers = num_coupling_layers
        self.coupling_layers = nn.ModuleList([
            AffineCoupling(in_features) for _ in range(num_coupling_layers)
        ])

    def forward(self, x, reverse=False):
        log_det_Jacobian = 0
        if not reverse:
            for coupling_layer in self.coupling_layers:
                x = coupling_layer(x)
                log_det_Jacobian += torch.sum(coupling_layer.scale_net(x[:, :in_features // 2]), dim=1)
        else:
            for coupling_layer in reversed(self.coupling_layers):
                x = coupling_layer(x, reverse=True)
        return x, log_det_Jacobian

############
# Define a function to visualize data transformation through the model
def visualize_transformation(model, data):
    # Convert data to tensor
    data = torch.FloatTensor(data)

    # Transform data through the model
    transformed_data, _ = model(data)

    return transformed_data.detach().numpy()


# Initialize the normalizing flow model with Kaiming He initialization for better convergence
def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')

num_coupling_layers = 20
in_features = 2
flow_model = NormalizingFlow(num_coupling_layers, in_features)
flow_model.apply(weights_init)  # Initialize weights

# Training
num_epochs = 1000
lr = 0.001
weight_decay = 1e-5  # L2 regularization strength to prevent overfitting
optimizer = optim.Adam(flow_model.parameters(), lr=lr, weight_decay=weight_decay)

for epoch in range(num_epochs):
    optimizer.zero_grad()
    z, log_det_Jacobian = flow_model(torch.FloatTensor(data))
    loss = -torch.mean(log_det_Jacobian)  # Negative log-likelihood
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item()}")

# Generate samples from the trained model
with torch.no_grad():
    z_samples = torch.randn(n_samples, in_features)
    x_samples, _ = flow_model(z_samples, reverse=True)
    x_samples = x_samples.numpy()

#%%
# Plot the original data and generated samples
plt.scatter(data[:, 0], data[:, 1], label='Original Data', alpha=0.5)
plt.scatter(x_samples[:, 0], x_samples[:, 1], label='Generated Samples', alpha=0.5)
plt.legend()
plt.show()


# Visualize the original data, generated samples, and the transformation
plt.figure(figsize=(12, 4))

# Plot original data
plt.subplot(131)
plt.scatter(data[:, 0], data[:, 1], label='Original Data', alpha=0.5)
plt.title("Original Data")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")

# Plot generated samples
plt.subplot(132)
plt.scatter(x_samples[:, 0], x_samples[:, 1], label='Generated Samples', alpha=0.5)
plt.title("Generated Samples")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")

# Plot transformation through the model
transformed_data = visualize_transformation(flow_model, data)
plt.subplot(133)
plt.scatter(transformed_data[:, 0], transformed_data[:, 1], label='Transformed Data', alpha=0.5)
plt.title("Transformation through Model")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")

plt.tight_layout()
plt.show()
# %%
