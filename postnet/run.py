
import torch
import numpy as np
import torch.optim as optim
from torchvision import transforms
import wandb

from LoadDataset_class import LoadDataset, load_Moons, load_MNIST, load_CIFAR
from PosteriorNetwork_class import NormalisingFlow, AffineCouplingLayer, Conditioner, PosteriorNetwork
from Encoder_Moons import Encoder_Moons
from Encoder_MNIST import Encoder_MNIST
from Encoder_CIFAR import Encoder_CIFAR
from Learning_scheduler import GradualWarmupScheduler, train
from evaluate import evaluate_model, image_show # computes test metrics


datasets = ['MNIST'] # ['Moons', 'MNIST', 'CIFAR'] 
models = ['PostNet'] #['PostNet', 'Ensemble']
m = 0 # indexing for the different models

#for Moons, MNIST, CIFAR respectively
latent_dim = [2, 10, 6] 
num_flows = [6,6,6] # number of coupling flow layers
training_lr = [5e-5, 5e-5, 5e-4] # start of training (end of warmup ) #Note: High LR create NaNs and Inf 
num_epochs = [200, 2, 100] # flere epochs maybe 12000
reg = [1e-5, 1e-5, 1e-5] # entropy regularisation

warmup_steps= [1000, 1000, 1000]
validation_every_steps = 50 
early_stop_patience = 12 # so after n-validations without improvement, stop training
early_stop_delta = 0.001 #in procent this is 0.1% 

num_classes = [2, 10, 10]
start_lr = 1e-9 # start of warmup
weight_decay = 5e-7  # L2 regularization strength to prevent overfitting in Adam or AdamW 
batch_size = 64

#data_dim = [2, 10, 6] # the same as latent dim # the encoder outputs n-D latent space
num_params = 2 # s and t
num_hidden = 3 # number of hidden layers
hidden_dim = 64 # neurons in hidden layers

annealing_interval = 40 # Every 10 epochs, anneal LR (warm restart)
min_lr = 1e-7 # during cosine annealing

split = lambda x: x.chunk(2, dim=-1)

dataset_name = 'MNIST'
ood_dataset_names = ['Fashion-MNIST', 'KMNIST']
subset_percentage = 1
split_ratios = [.6, .8]

seed = 123
torch.manual_seed(seed)
np.random.seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

wandb.init(
    project='Normalising-Flow-DNN',
    config={
        'architecture': models[0],
        'dataset': dataset_name,
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
        'num_classes': num_classes,
        'latent_dim': latent_dim,
        'num_params': num_params,
        'num_hidden': num_hidden,
        'hidden_dim': hidden_dim,
        'num_flows': num_flows,
        'split': split,

    }
    #name='run_name',
    #tags=['experiment1', ''],
    #notes='Trying out a new architecture',
    #dir='/path/to/log/files',
    ##entity='my_team',
    #group='experiment_group',
    #job_type='train'
)

# Run script to test Postnet model on all dataset
# - Moons: Uses Encoder_Moons.py, latent dim = 2, num_flows = 6
# - MNIST: Uses Encoder_MNIST.py, latent dim = 10, num_flows = 6
# - CIFAR: Uses Encoder_CIFAR.py, latent dim = 6, num_flows = 6
#for model_name in models: # PostNet, Ensemble

# Load dataset
i=0 # indexing for the different datasets
for dataset_name in datasets: # Moons, MNIST, CIFAR
    print(f'Running {models[0]} on {dataset_name} dataset')

    if dataset_name == 'TwoMoons':
        i=1
        loaders, N_counts, set_lengths, ood_dataset_loaders, N_ood = load_Moons(batch_size, subset_percentage=None, split_ratios=[0.6,0.8], seed=seed)

    elif dataset_name == 'MNIST':
        i=2
        loaders, N_counts, set_lengths, ood_dataset_loaders, N_ood = load_MNIST(batch_size, subset_percentage=None, split_ratios=[0.6,0.8], seed=seed)
            
    elif dataset_name == 'CIFAR':
        i=3
        # Load CIFAR dataset
        loaders, N_counts, set_lengths, ood_dataset_loaders, N_ood = load_CIFAR(batch_size, subset_percentage=None, split_ratios=[0.6,0.8], seed=seed)
    
# Instantiate model
flow_models = []
for class_label in range(num_classes[i]):
    conditioner = Conditioner(in_dim=latent_dim[i]//2, out_dim=latent_dim[i]//2, num_hidden=num_hidden, hidden_dim=hidden_dim, num_params=num_params)
    affine_coupling = AffineCouplingLayer(conditioner, split, latent_dim[i]) # split the tensor into 2 parts
    flows = [affine_coupling for _ in range(num_flows[i])]
    latent_distribution = torch.distributions.MultivariateNormal(loc=torch.zeros(latent_dim[i]).to(device), scale_tril=torch.eye(latent_dim[i]).to(device))
    flow_model = NormalisingFlow(latent_distribution, flows).to(device)
    #flow_model = NormalisingFlow(latent_distribution, flows).apply(init_weights).to(device)
    flow_models.append(flow_model)
postnet_model = PosteriorNetwork(latent_dim[i], flow_models, N_counts['train'], num_classes[i], reg[i], dataset_name).to(device) 
optimiser = optim.AdamW(postnet_model.parameters(), lr=training_lr[i], weight_decay=weight_decay)
warmup_scheduler = GradualWarmupScheduler(optimiser, warmup_steps=warmup_steps[i], start_lr=start_lr, end_lr=training_lr[i])
training_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=annealing_interval, eta_min=min_lr, last_epoch=-1)

# string with the name of the model and the parameters used (must work for linux as well)

save_model = f"{dataset_name}\\{models[m]}_REG{reg[i]}_LR{training_lr[i]}_NF{num_flows[i]}_Warm{warmup_steps[i]}_Epoch{num_epochs[i]}_WD{weight_decay}_Hid{hidden_dim}_hLay{num_hidden}_LD{latent_dim[i]}"
#print(save_model)

train_losses, val_losses, train_accuracies, val_accuracies, all_train_losses = train(postnet_model, optimiser, loaders['train'], loaders['val'], 
                                        num_epochs[i], validation_every_steps, early_stop_delta, early_stop_patience, warmup_scheduler, 
                                        training_scheduler, warmup_steps[i], N_counts, set_lengths, device, save_model)        


