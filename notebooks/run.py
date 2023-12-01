

#datasets = ['TwoMoons', 'MNIST', 'CIFAR']
datasets = ['MNIST']
#models = ['PostNet', 'Ensemble']
models = ['PostNet']
entropy = 1e-5

#for TwoMoons, MNIST, CIFAR respectively

latent_dim = [2, 10, 6] 
num_flows = [6,6,6] # number of coupling flow layers
training_lr = [5e-5, 5e-5, 5e-4] # start of training (end of warmup ) #Note: High LR create NaNs and Inf 
num_epochs = [200, 100, 100] # flere epochs maybe 12000
reg = [1e-5, 1e-5, 1e-5] # entropy regularisation

warmup_steps= [1000, 1000, 1000]
validation_every_steps = 50 
early_stop_patience = 12 # so after n-validations without improvement, stop training
early_stop_delta = 0.001 #in procent this is 0.1% 

num_classes = [2, 10, 10]
start_lr = 1e-9 # start of warmup
weight_decay = 5e-7  # L2 regularization strength to prevent overfitting in Adam or AdamW 
batch_size = 64

data_dim = [2, 10, 6] # the same as latent dim # the encoder outputs n-D latent space
in_dim= data_dim // 2 # since we split the data
out_dim= data_dim // 2
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


