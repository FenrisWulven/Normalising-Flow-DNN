import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataset import Subset
from torchvision import datasets, transforms
from sklearn.datasets import make_moons, make_circles, make_blobs

import os
# append rood dir called Normalising-Flow-DNN to sys.path
get_cwd = os.getcwd()
root_dir = os.path.dirname(get_cwd)
data_dir = os.path.join(root_dir, 'data')

class LoadDataset:
    def __init__(self, dataset_name, root_dir, train, transform, subset_percentage=None, split_ratios=[0.6,0.8], seed=None, n_samples=3000, noise=0.1):
        if dataset_name == 'Moons':
            X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
            self.dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
        elif dataset_name == 'Circles':
            X_c, y_c = make_circles(n_samples=n_samples, noise=0.01, factor=0.1, random_state=seed)
            np.random.seed(seed)
            outer_circle_noise = np.random.normal(scale=0.1, size=X_c[y_c == 0].shape) 
            inner_circle_noise = np.random.normal(scale=0.4, size=X_c[y_c == 1].shape) 
            X_c[y_c == 0] += outer_circle_noise # Add custom noise to outer circle
            X_c[y_c == 1] += inner_circle_noise
            X_c[y_c == 0] *= 4  # Scale big circle radius
            X_c[y_c == 1] *= 0.23  # Scale small circle radius
            X_c[y_c == 1] += [0.5, 0.25]  # Translate small circle to center
            self.dataset = TensorDataset(torch.tensor(X_c, dtype=torch.float32), torch.tensor(y_c, dtype=torch.long))
        elif dataset_name == 'Disk':
            X_c, y_c = make_circles(n_samples=n_samples, noise=0.01, factor=0.8, random_state=seed)
            np.random.seed(seed)
            outer_circle_noise = np.random.normal(scale=0.6, size=X_c[y_c == 0].shape) 
            inner_circle_noise = np.random.normal(scale=0.5, size=X_c[y_c == 1].shape) 
            X_c[y_c == 0] += outer_circle_noise # Add custom noise to outer circle
            X_c[y_c == 1] += inner_circle_noise
            X_c[y_c == 0] *= 0.15  # Scale big circle radius
            X_c[y_c == 1] *= 0.15  # Scale small circle radius
            X_c[y_c == 0] += [1.45, 0.45]  # Translate larger circle to center
            X_c[y_c == 1] += [0.5, 0.25]  # Translate small circle to center
            self.dataset = TensorDataset(torch.tensor(X_c, dtype=torch.float32), torch.tensor(y_c, dtype=torch.long))
        elif dataset_name == 'Blobs':
            X, y = make_blobs(n_samples=n_samples, centers=[[-3,3],[4,-2]], n_features=2, cluster_std=1.0, random_state=seed)
            self.dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
        elif dataset_name == 'SVHN':  #use torchvision built-in datasets
            self.dataset = datasets.SVHN(root=root_dir, split='test', download=True, transform=transform)
        else: 
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

    def get_data_loaders(self, batch_size, split_ratios, num_workers=4, pin_memory=True):
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

        return train_loader, val_loader, test_loader

    def get_ood_data_loaders(self, batch_size, num_workers=4, pin_memory=True):
        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        ood_split = 0.5
        num_samples = len(self.dataset)
        indices = list(range(num_samples))
        split = int(np.floor(ood_split* num_samples))

        np.random.shuffle(indices)
        ood_val_idx, ood_test_idx = indices[:split], indices[split:]
        self.ood_val_indices = ood_val_idx
        self.ood_test_indices = ood_test_idx 

        val_sampler = SubsetRandomSampler(ood_val_idx)
        test_sampler = SubsetRandomSampler(ood_test_idx)

        val_loader = DataLoader(self.dataset, batch_size=batch_size, sampler=val_sampler, num_workers=num_workers, pin_memory=pin_memory)
        test_loader = DataLoader(self.dataset, batch_size=batch_size, sampler=test_sampler, num_workers=num_workers, pin_memory=pin_memory)

        return val_loader, test_loader

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

        if isinstance(original_dataset, TensorDataset):
        # Extract labels from the second element of the tensors in the dataset
            labels = original_dataset.tensors[1].numpy()
        elif isinstance(original_dataset, datasets.SVHN):
            labels = original_dataset.labels
        else:
            labels = original_dataset.targets

        num_classes = len(np.unique(labels))
        class_counts = np.zeros(num_classes)
        for idx in indices:
            if idx < len(labels):
                target = labels[idx]
                class_counts[target] += 1
            else:
                raise IndexError(f"Index {idx} is out of bounds for dataset with size {len(labels)}")

        return torch.tensor(class_counts)

    def get_set_length(self, data_loader):
        return sum(len(batch) for batch, _ in data_loader)


def load_Moons(batch_size, n_samples=3000, noise=0.1, split_ratios=[0.6,0.8], seed=None):
    dataset = LoadDataset('Moons', None, train=True, transform=None, n_samples=n_samples, noise=noise, split_ratios=split_ratios, seed=seed)
    loaders = {}
    loaders['train'], loaders['val'], loaders['test'] = dataset.get_data_loaders(batch_size=batch_size, split_ratios=split_ratios)
    
    N_counts = {}
    N_counts['train'] = dataset.get_class_counts(dataset.train_indices)
    N_counts['val'] = dataset.get_class_counts(dataset.val_indices)
    N_counts['test'] = dataset.get_class_counts(dataset.test_indices)

    set_lengths = {}
    set_lengths['train'] = dataset.get_set_length(loaders['train'])
    set_lengths['val'] = dataset.get_set_length(loaders['val'])
    set_lengths['test'] = dataset.get_set_length(loaders['test'])

    ood_dataset_names = ['Disk', 'Circles', 'Blobs'] #Disk
    ood_dataset_loaders = {'val': {}, 'test': {}}
    N_ood = {'val': {}, 'test': {}}

    for ood_dataset_name in ood_dataset_names: 
        ood_dataset = LoadDataset(ood_dataset_name, None, train=False, transform=None, n_samples=n_samples, noise=noise, split_ratios=split_ratios, seed=seed)
        ood_dataset_loaders['val'][ood_dataset_name], ood_dataset_loaders['test'][ood_dataset_name] = ood_dataset.get_ood_data_loaders(batch_size=batch_size)
        N_ood['val'][ood_dataset_name] = ood_dataset.get_class_counts(ood_dataset.ood_val_indices)
        N_ood['test'][ood_dataset_name] = ood_dataset.get_class_counts(ood_dataset.ood_test_indices)

    return loaders, N_counts, set_lengths, ood_dataset_loaders, N_ood


def load_MNIST(batch_size, subset_percentage=None, ood_subset_percentage = None, split_ratios=[0.6,0.8], seed=None):
    transform = transforms.Compose([ transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)) ]) # makes MNIST pixels [-1, 1]
    dataset = LoadDataset('MNIST', data_dir, train=True, transform=transform, subset_percentage=subset_percentage, split_ratios=split_ratios, seed=seed)
    loaders = {}
    loaders['train'], loaders['val'], loaders['test'] = dataset.get_data_loaders(batch_size=batch_size, split_ratios=split_ratios)
    
    N_counts = {}
    N_counts['train'] = dataset.get_class_counts(dataset.train_indices)
    N_counts['val'] = dataset.get_class_counts(dataset.val_indices)
    N_counts['test'] = dataset.get_class_counts(dataset.test_indices)

    set_lengths = {}
    set_lengths['train'] = dataset.get_set_length(loaders['train'])
    set_lengths['val'] = dataset.get_set_length(loaders['val'])
    set_lengths['test'] = dataset.get_set_length(loaders['test'])

    ood_dataset_names = ['FashionMNIST', 'KMNIST']
    ood_dataset_loaders = {'val': {}, 'test': {}}
    N_ood = {'val': {}, 'test': {}}

    ood_transform = {}
    ood_transform['FashionMNIST'] = transforms.Compose([ transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,)) ])
    ood_transform['KMNIST'] = transforms.Compose([ transforms.ToTensor(), transforms.Normalize((0.1904,), (0.3475,)) ])

    for ood_dataset_name in ood_dataset_names: 
        ood_dataset = LoadDataset(ood_dataset_name, data_dir, train=False, transform=ood_transform[ood_dataset_name], subset_percentage=ood_subset_percentage, split_ratios=split_ratios, seed=seed)
        ood_dataset_loaders['val'][ood_dataset_name], ood_dataset_loaders['test'][ood_dataset_name] = ood_dataset.get_ood_data_loaders(batch_size=batch_size)
        N_ood['val'][ood_dataset_name] = ood_dataset.get_class_counts(ood_dataset.ood_val_indices)
        N_ood['test'][ood_dataset_name] = ood_dataset.get_class_counts(ood_dataset.ood_test_indices)

    return loaders, N_counts, set_lengths, ood_dataset_loaders, N_ood


def load_CIFAR(batch_size, subset_percentage=None, ood_subset_percentage = None, split_ratios=[0.6,0.8], seed=None):
    transform = transforms.Compose([ transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    dataset = LoadDataset('CIFAR10', data_dir, train=True, transform=transform, subset_percentage=subset_percentage, split_ratios=split_ratios, seed=seed)
    loaders = {}
    loaders['train'], loaders['val'], loaders['test'] = dataset.get_data_loaders(batch_size=batch_size, split_ratios=split_ratios)
    
    N_counts = {}
    N_counts['train'] = dataset.get_class_counts(dataset.train_indices)
    N_counts['val'] = dataset.get_class_counts(dataset.val_indices)
    N_counts['test'] = dataset.get_class_counts(dataset.test_indices)

    set_lengths = {}
    set_lengths['train'] = dataset.get_set_length(loaders['train'])
    set_lengths['val'] = dataset.get_set_length(loaders['val'])
    set_lengths['test'] = dataset.get_set_length(loaders['test'])

    ood_dataset_names = ['SVHN']
    ood_dataset_loaders = {'val': {}, 'test': {}}
    N_ood = {'val': {}, 'test': {}}

    ood_transform = {}
    ood_transform['SVHN'] = transforms.Compose([ transforms.ToTensor(), transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)) ])
    for ood_dataset_name in ood_dataset_names: 
        ood_dataset = LoadDataset(ood_dataset_name, data_dir, train=False, transform=ood_transform[ood_dataset_name], subset_percentage=ood_subset_percentage, split_ratios=split_ratios, seed=seed)
        ood_dataset_loaders['val'][ood_dataset_name], ood_dataset_loaders['test'][ood_dataset_name] = ood_dataset.get_ood_data_loaders(batch_size=batch_size)
        N_ood['val'][ood_dataset_name] = ood_dataset.get_class_counts(ood_dataset.ood_val_indices)
        N_ood['test'][ood_dataset_name] = ood_dataset.get_class_counts(ood_dataset.ood_test_indices)

    return loaders, N_counts, set_lengths, ood_dataset_loaders, N_ood

