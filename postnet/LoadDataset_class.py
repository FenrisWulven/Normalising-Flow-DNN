from typing import Any
import torch
import numpy as np
from torch.utils.data import DataLoader #Dataset, 
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataset import Subset
from torchvision import datasets, transforms
import os
import sys
# append rood dir called Normalising-Flow-DNN to sys.path
get_cwd = os.getcwd()
root_dir = os.path.dirname(get_cwd)
data_dir = os.path.join(root_dir, 'data')

class LoadDataset:
    def __init__(self, dataset_name, root_dir, train, transform, subset_percentage=None, split_ratios=[0.6,0.8], seed=None):
        #use torchvision datasets
        self.dataset = datasets.__dict__[dataset_name](root=root_dir, train=train, download=True, transform=transform)
        self.subset_percentage = subset_percentage
        self.split_ratios = split_ratios
        self.seed = seed
        if self.subset_percentage is not None:
            self._create_subset()

    def _create_subset(self):
        num_samples = int(len(self.dataset) * self.subset_percentage)
        if self.seed is not None:
            np.random.seed(self.seed)
        indices = np.random.permutation(len(self.dataset))[:num_samples]
        self.dataset = Subset(self.dataset, indices)

    def get_data_loaders(self, batch_size, split_ratios, num_workers=0, pin_memory=True):
        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        num_train = len(self.dataset)
        indices = list(range(num_train))
        split1 = int(np.floor(split_ratios[0] * num_train))
        split2 = int(np.floor(split_ratios[1] * num_train))

        np.random.shuffle(indices)
        train_idx, val_idx, test_idx = indices[:split1], indices[split1:split2], indices[split2:]
        self.train_indices = train_idx
        self.val_indices = val_idx
        self.test_indices = test_idx

        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        test_sampler = SubsetRandomSampler(test_idx)

        train_loader = DataLoader(self.dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=pin_memory)
        val_loader = DataLoader(self.dataset, batch_size=batch_size, sampler=val_sampler, num_workers=num_workers, pin_memory=pin_memory)
        test_loader = DataLoader(self.dataset, batch_size=batch_size, sampler=test_sampler, num_workers=num_workers, pin_memory=pin_memory)

        # Calculate class counts for the training set
        # if isinstance(self.dataset, Subset):
        #     original_dataset = self.dataset.dataset
        # else:
        #     original_dataset = self.dataset

        # class_counts = np.zeros(len(original_dataset.classes))
        # for idx in train_idx:
        #     _, target = original_dataset[idx]
        #     class_counts[target] += 1
        # N = torch.tensor(class_counts)

        return train_loader, val_loader, test_loader
    # Changed
        # class_index, class_count = np.unique(self.Y[:int(split_ratios[0] * num_train)], return_counts=True)
        # N = np.zeros(self.output_dim)
        # N[class_index.astype(int)] = class_count
        # N = torch.tensor(N)

    def get_full_loader(self, batch_size, num_workers=0, pin_memory=True):
        # Create a DataLoader for the entire dataset without splitting
        full_loader = DataLoader(self.dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True)
        self.indices = list(range(len(self.dataset)))
        return full_loader

    def get_class_counts(self, indices):
        if isinstance(self.dataset, Subset):
            original_dataset = self.dataset.dataset
        else:
            original_dataset = self.dataset

        class_counts = np.zeros(len(original_dataset.classes))
        for idx in indices:
            target = original_dataset.targets[idx]
            class_counts[target] += 1
        return torch.tensor(class_counts)

        # if isinstance(data_loader.dataset, Subset):
        #     # If the data loader is using a subset of the dataset
        #     original_dataset = data_loader.dataset.dataset
        #     indices = data_loader.dataset.indices
        #     class_counts = np.zeros(len(original_dataset.classes))
        #     for idx in indices:
        #         target = original_dataset.targets[idx]
        #         class_counts[target] += 1
        # else:
        #     # If the data loader is using the full dataset
        #     original_dataset = data_loader.dataset
        #     class_counts = np.zeros(len(original_dataset.classes))
        #     for _, target in original_dataset:
        #         class_counts[target] += 1
        # return torch.tensor(class_counts)
    def get_set_length(self, data_loader):
        return sum(len(batch) for batch, _ in data_loader)


def load_Moons(batch_size, subset_percentage=None, split_ratios=[0.6,0.8], seed=None):
    # transform = transforms.Compose([transforms.ToTensor()])
    # dataset = LoadDataset(dataset_name=dataset_name, root_dir='./data', train=True, transform=transform, subset_percentage=subset_percentage, split_ratios=split_ratios, seed=seed)
    # train_loader, val_loader, test_loader, N = dataset.get_data_loaders(batch_size=batch_size, split_ratios=split_ratios)
    # ood_dataset_loaders = {}
    # ood_N = {}
    return True


def load_MNIST(batch_size, subset_percentage=None, ood_subset_percentage = None, split_ratios=[0.6,0.8], seed=None):
    transform = transforms.Compose([ transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)) ]) # makes MNIST pixels [-1, 1]
    dataset = LoadDataset('MNIST', data_dir, train=True, transform=transform, subset_percentage=subset_percentage, split_ratios=split_ratios, seed=seed)
    loaders = {}
    loaders['train'], loaders['val'], loaders['test'] = dataset.get_data_loaders(batch_size=batch_size, split_ratios=split_ratios)
    
    #y_train = loaders['train'].dataset.dataset.targets
    N_counts = {}
    N_counts['train'] = dataset.get_class_counts(dataset.train_indices)
    N_counts['val'] = dataset.get_class_counts(dataset.val_indices)
    N_counts['test'] = dataset.get_class_counts(dataset.test_indices)

    set_lengths = {}
    set_lengths['train'] = dataset.get_set_length(loaders['train'])
    set_lengths['val'] = dataset.get_set_length(loaders['val'])
    set_lengths['test'] = dataset.get_set_length(loaders['test'])

    ood_dataset_names = ['FashionMNIST', 'KMNIST']
    ood_dataset_loaders = {}
    N_ood = {}
    for ood_dataset_name in ood_dataset_names: 
        ood_dataset = LoadDataset(ood_dataset_name, data_dir, train=False, transform=transform, subset_percentage=ood_subset_percentage, split_ratios=split_ratios, seed=seed)
        ood_dataset_loaders[ood_dataset_name] = ood_dataset.get_full_loader(batch_size=batch_size)
        N_ood[ood_dataset_name] = ood_dataset.get_class_counts(ood_dataset.indices)

    return loaders, N_counts, set_lengths, ood_dataset_loaders, N_ood


def load_CIFAR():
    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    # dataset = LoadDataset(dataset_name=dataset_name, root_dir='./data', train=True, transform=transform, subset_percentage=subset_percentage, split_ratios=split_ratios, seed=seed)
    # train_loader, val_loader, test_loader, N = dataset.get_data_loaders(batch_size=batch_size, split_ratios=split_ratios)
    # ood_dataset_loaders = {}
    # ood_N = {}
    # for ood_dataset_name in ood_dataset_names:
    return True

