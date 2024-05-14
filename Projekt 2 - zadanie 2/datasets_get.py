import os
import torch 
import numpy as np
import pandas as pd
from sklearn import datasets as sklearn_datasets
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from torchvision import datasets, transforms
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from joblib import dump, load
import torchvision


class CustomDataset(Dataset):
    def __init__(self, data, targets, device):
        self.data = torch.tensor(data, dtype=torch.double, device=device)
        self.targets = torch.tensor(targets, dtype=torch.long, device=device)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = {'data': self.data[idx], 'target': self.targets[idx]}
        return sample

transform_mnist = transforms.Compose([
        transforms.ToTensor()
    ])
transform_cifar10_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),     # Losowe odwrócenie obrazu w poziomie
    transforms.RandomRotation(10),          # Losowe obrócenie obrazu o maksymalnie 10 stopni
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Zmiana jasności, kontrastu, nasycenia i barwy
    transforms.RandomCrop(32, padding=4),   # Losowe przycięcie obrazu
    transforms.ToTensor(),                  # Zamiana obrazu PIL na tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalizacja obrazu
])

# Definicja transformacji dla danych testowych
transform_cifar10_test = transforms.Compose([
    transforms.ToTensor(),                  # Zamiana obrazu PIL na tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalizacja obrazu
])


def mnist_flatten(device, train):
    mnist           = datasets.MNIST(root='data', train=train, download=True, transform=transform_mnist)
    flattened_mnist = mnist.data.flatten(start_dim=1)
    mnists          = CustomDataset(data=StandardScaler().fit_transform(flattened_mnist), targets=mnist.targets, device=device)
    return mnists, 28*28, 10, 'mnist_flatten', 784


def mnist_to_cnn(device, train):
    mnist           = datasets.MNIST(root='data', train=train, download=True, transform=transform_mnist)
    mnists          = CustomDataset(data=mnist.data, targets=mnist.targets, device=device)
    mnists.data     = mnists.data.view(-1,1,28,28)
    return mnists

def cifar10_to_cnn(device, train):
    if train is False:
        cifar           = torchvision.datasets.CIFAR10(root='./data', train=train, download=True, transform=transform_cifar10_test)
        cifars          = CustomDataset(data=cifar.data, targets=cifar.targets, device=device)
        print(cifars.data.size())
        # cifars.data     = cifars.data.reshape(-1,32,32,3)
        cifars.data     = torch.permute(cifars.data, (0, 3, 1, 2))
        print(cifars.data.size())
    else:
        cifar           = torchvision.datasets.CIFAR10(root='./data', train=train, download=True, transform=transform_cifar10_train)
        cifars          = CustomDataset(data=cifar.data, targets=cifar.targets, device=device)
        print(cifars.data.size())
        # cifars.data     = cifars.data.reshape(-1,32,32,3)
        cifars.data     = torch.permute(cifars.data, (0, 3, 1, 2))
        print(cifars.data.size())

    return cifars
