
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.nn.utils as utils
import torch.optim.lr_scheduler as lr_scheduler
from torch.distributions import Distribution
from torch.distributions.dirichlet import Dirichlet
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import math
from typing import Tuple, List, Callable

from sklearn.preprocessing import StandardScaler
from torchvision import datasets, transforms
from torchvision import utils as vutils
from sklearn import metrics
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split
from scipy.stats import gaussian_kde
from IPython.display import clear_output

import wandb
import os 
os.environ['WANDB_NOTEBOOK_NAME'] = 'Coupling_mnist_plots.ipynb'
# set seed
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

############ HYPERPARAMETERS ################
#lr = 0.00005
training_lr = 0.0005 # start of training (end of warmup ) #Note: High LR create NaNs and Inf 
start_lr = 1e-9 # start of warmup
min_lr = 1e-7 # during cosine annealing
num_epochs = 200 # flere epochs maybe 12000
warmup_steps= 2000
validation_every_steps = 500 # is actually every epoch in training loop!!
#validation_every_epochs = 1
weight_decay = 5e-7  # L2 regularization strength to prevent overfitting in Adam or AdamW 
batch_size = 64
early_stop_delta = 0.001 #in procent this is 0.1% 
early_stop_patience = 20 # so after 20 validations without improvement, stop training
reg = 1e-6 # entropy regularisation
annealing_interval = 40 # Every 10 epochs, anneal LR (warm restart)

num_ensemble_models = 10 # ændre til 10  # Define the number of models in the ensemble
num_classes = 10
latent_dim = 6 # Change to 4 or 6    # the encoder outputs 2D latent space
# data_dim = 6 # the encoder outputs 2D latent space
# in_dim= data_dim // 2 # since we split the data
# out_dim= data_dim // 2
# num_params = 2 # s and t
# num_hidden = 3 # number of hidden layers
# hidden_dim = 64 # neurons in hidden layers

wandb.init(
    project='Normalising-Flow-DNN',
    config={
        'architecture': 'Ensemble',
        'dataset': 'MNIST',
        'training_lr': training_lr,
        'start_lr': start_lr,
        'min_lr': min_lr,
        'num_epochs': num_epochs,
        'warmup_steps': warmup_steps,
        'validation_every_steps': validation_every_steps,
        'weight_decay': weight_decay,
        'batch_size': batch_size,
        'early_stop_delta': early_stop_delta,
        'early_stop_patience': early_stop_patience,
        'reg': reg,
        'annealing_interval': annealing_interval,
        'num_ensemble_models': num_ensemble_models,
        'num_classes': num_classes,
        'latent_dim': latent_dim,
    }
    #name='run_name',
    #tags=['experiment1', ''],
    #notes='Trying out a new architecture',
    #dir='/path/to/log/files',
    ##entity='my_team',
    #group='experiment_group',
    #job_type='train'
)


############ CLASSES ################
# Define the Normalising Flow model template

class CNN(nn.Module):

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

class Ensemble(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super(Ensemble, self).__init__()
        self.cnn = CNN(latent_dim)
        self.fc = nn.Linear(latent_dim, num_classes)

    def forward(self, x):
        features = self.cnn(x)
        output = self.fc(features)
        return output

class GradualWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, start_lr, end_lr, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.lr_increment = (end_lr - start_lr) / warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            lr = self.start_lr + self.last_epoch * self.lr_increment
            return [lr for _ in self.base_lrs]
        return self.base_lrs
    
############ LOAD MNIST DATASET ################
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
batch_size = 64 # standard value
from torch.utils.data.dataset import Subset

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
#subset_percentage = 0.1
#num_samples = int(len(train_dataset) * subset_percentage)
#print("Number of samples:", num_samples)
num_train = int(len(train_dataset) * 0.5)
num_test = int(len(test_dataset) * 0.5)
print("n_train", num_train, "n_test", num_test)
train_subset = Subset(train_dataset, range(num_train))
test_subset = Subset(test_dataset, range(num_test)) 

train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)

# Get ground-truth label counts N_c
# Dictionary with class names from indexes.
classes = {index: name for name, index in train_dataset.class_to_idx.items()}

# Initialise dictionary to store the counts for each class using class indexes
N = {index: 0 for index in range(len(classes))}
# Count the occurrences of each class
#for _, target in train_dataset:
for _, target in train_subset:
    N[target] += 1
N = torch.tensor([N[index] for index in range(len(classes))])
print("N:", N)

#y_train = torch.tensor([target for _, target in train_dataset])
y_train = torch.tensor([target for _, target in train_subset])

############# FUNCTION ##################
def image_show(img):
    img = img.detach().cpu()
    img = img / 2 + 0.5   #Unnormalise
    with sns.axes_style("white"):
        plt.figure(figsize=(8, 8))
        plt.imshow(img.permute((1, 2, 0)).numpy())
        plt.axis('off')
        plt.show()

def init_weights(model):
    if isinstance(model, nn.Linear):
        nn.init.xavier_normal_(model.weight)
        nn.init.zeros_(model.bias) 

def accuracy(y_train, preds):
    accuracy = accuracy_score(y_train.cpu().numpy(), preds.cpu().numpy())
    return accuracy

############# INSTANTIATE MODEL ##################
ensemble_models = [Ensemble(latent_dim, num_classes).to(device) for _ in range(num_ensemble_models)]
optimisers = [optim.AdamW(model.parameters(), lr=training_lr, weight_decay=weight_decay) for model in ensemble_models]

criterion = nn.CrossEntropyLoss()

total_steps_per_epoch = len(train_loader)  # Total batches (steps) per epoch
warmup_epochs = math.ceil(warmup_steps / total_steps_per_epoch)  # Total warmup epochs

############## TRAINING ######################
# Trains one model in the ensemble
ensemble_number = 0
def train(model, optimiser, train_loader, test_loader, num_epochs, validation_every_steps, 
          early_stop_delta, early_stop_patience, warmup_scheduler, training_scheduler, warmup_steps, ensemble_number):
    model.train()
    train_losses, train_accuracies, val_losses, val_accuracies = [], [], [], []
    all_train_losses = []
    best_val_loss = float("Inf")
    step = 0 # how many batches we have trained on (each batch is 64 samples) #9000 training samples / 64 batch size = 140 batches per epoch
    counter = 0 # for early stopping 
    early_stopping = False
    wandb.watch(model, log="all")

    for epoch in range(num_epochs): #epoch is one forward pass through the entire training set
        train_losses_batches, train_accuracies_batches = [], []
        #batches_counter = 0

        for batch_index, (X_train, y_train) in enumerate(train_loader):
            X_train, y_train = X_train.to(device), y_train.to(device)
            # batches_counter += 1
            # Forward pass
            output = model(X_train)
            loss = criterion(output, y_train) #CrossEntropy loss
            # Perform one training step
            optimiser.zero_grad()
            loss.backward()
            utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimiser.step()
            if step < warmup_steps:
                warmup_scheduler.step()         
            step += 1
            #train_losses.append(loss.item())
            # Compute training accuracy and loss for this batch
            with torch.no_grad():
                preds = torch.argmax(output, dim=1)
                train_accuracy_batch = accuracy(y_train, preds)
                train_accuracies_batches.append(train_accuracy_batch)
                train_losses_batches.append(loss.item())
                all_train_losses.append(loss.item())
                current_lr = optimiser.param_groups[0]['lr']
                wandb.log({"batch_train_losses": loss.item(), "batch_train_accuracy": 
                           train_accuracy_batch, "step": step, "learning_rate": current_lr, "epoch": epoch})
                
            if step % validation_every_steps == 0:
                train_loss = np.mean(train_losses_batches)
                train_losses.append(train_loss)
                train_accuracy = np.mean(train_accuracies_batches)
                train_accuracies.append(train_accuracy)
                wandb.log({"train_loss": train_loss, "train_accuracy": train_accuracy, "step": step})

                val_losses_batches = []
                #val_accuracies_batches = []
                val_correct = []
                model.eval()
                with torch.no_grad():   
                    for batch_index, (X_test, y_test) in enumerate(test_loader):
                        X_test, y_test = X_test.to(device), y_test.to(device)
                        # Evaluation Forward pass
                        output = model(X_test)
                        loss = criterion(output, y_test) #CrossEntropy loss
                        preds = torch.argmax(output, dim=1)
                        correct_batch = (preds == y_test).sum().item()
                        val_correct.append(correct_batch)
                        val_losses_batches.append(loss.item())

                val_accuracy = sum(val_correct) / len(test_subset) # or use len(test_dataset)
                #Multiply by len(test_dataset) because the final batch of DataLoader may be smaller (drop_last=False).
                val_accuracies.append(val_accuracy)
                val_loss = np.mean(val_losses_batches) 
                val_losses.append(val_loss)
                wandb.log({"val_loss": val_loss, "val_accuracy": val_accuracy, "step": step})
                model.train()

                if val_losses[-1] < -1.:
                    print("Unstable training")
                    break
                if np.isnan(val_losses[-1]):
                    print('Detected NaN Loss')
                    break
                # If val_loss is the best so far, save the model state_dict and reset the early stopping counter
                if val_losses[-1] < best_val_loss:
                    best_val_loss = val_losses[-1]
                    counter = 0
                    best_model = model.state_dict()
                    torch.save({'epoch': epoch, 'model_state_dict': best_model, 'loss': best_val_loss}, f'best_model_ensemble{ensemble_number}.pth')
                    print('Model saved')

                # Early stopping - if val_loss is not improving (plus a delta e-4 as buffer) then start counter
                # after patience of a certain number of validations, then stop training
                elif val_losses[-1] > (best_val_loss + early_stop_delta):
                    counter += 1
                    if counter >= early_stop_patience:
                        #print("Early stopping")
                        early_stopping = True
                        break
                
                print(f"Model: {ensemble_number}, Step: {step}, Epoch: {epoch+1}\tTrain Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Train Accuracy: {train_accuracies[-1]:.4f}, Val Accuracy: {val_accuracies[-1]:.4f}")
                #### Lave plots med meshgrid f-funktion af normalising flow undervejs for at se ændringen
        
        # Update training scheduler (annealing LR)
        if epoch >= warmup_epochs:
                training_scheduler.step()

        if early_stopping: # if true
            print("Early stopping triggered. Exiting training.")
            break  # Break out of the outer loop
    print("Finished training.")
    return train_losses, val_losses, train_accuracies, val_accuracies, all_train_losses #,model

# Train each model in the ensemble with warmup and annealing'
for model, optimiser in zip(ensemble_models, optimisers):
    warmup_scheduler = GradualWarmupScheduler(optimiser, warmup_steps=warmup_steps, start_lr=start_lr, end_lr=training_lr)
    training_scheduler = lr_scheduler.CosineAnnealingLR(optimiser, T_max=annealing_interval, eta_min=min_lr, last_epoch=-1)
    #training_scheduler= torch.optim.lr_scheduler.StepLR(optimiser, step_size=5, gamma=0.1)
    #training_scheduler= ExponentialLR(optimizer=default_optimizer, gamma=0.98)
    train_losses, val_losses, train_accuracies, val_accuracies, all_train_losses = train(model, optimiser, train_loader, test_loader, num_epochs, 
                                 validation_every_steps, early_stop_delta, early_stop_patience, warmup_scheduler, training_scheduler, warmup_steps, ensemble_number)
    ensemble_number += 1

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


############ PLOTS ################
# Plot loss of training and validation
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(train_losses, label='Train Loss')
axes[0].plot(val_losses, label='Validation Loss')
axes[0].set_xlabel('Validation Epochs')
axes[0].set_ylabel('Loss')
axes[0].set_title('Train and Validation Loss')
axes[0].legend()

# Plot accuracies of training and validation
axes[1].plot(train_accuracies, label='Train Accuracy')
axes[1].plot(val_accuracies, label='Validation Accuracy')
axes[1].set_xlabel('Steps')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Train and Validation Accuracy')
axes[1].legend()
plt.tight_layout()
plt.savefig('plots_mnist/training_loss_acc.png', bbox_inches='tight')

# plot all_train_losses
plt.figure(figsize=(12,8))
plt.plot(all_train_losses,  '.',label='Train Loss', alpha=0.3)
#plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Train Loss')
plt.legend()
plt.savefig('plots_mnist/training_all_losses_acc.png', bbox_inches='tight')


