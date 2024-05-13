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
transform_cifar10 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])



def mnist_flatten(device, train):
    mnist           = datasets.MNIST(root='data', train=train, download=True, transform=transform_mnist)
    flattened_mnist = mnist.data.flatten(start_dim=1)
    mnists          = CustomDataset(data=StandardScaler().fit_transform(flattened_mnist), targets=mnist.targets, device=device)
    return mnists, 28*28, 10, 'mnist_flatten', 784


def mnist_to_cnn(device, train):
    mnist           = datasets.MNIST(root='data', train=train, download=True, transform=transform_mnist)
    mnists          = CustomDataset(data=mnist.data, targets=mnist.targets, device=device)
    return mnists, 1,  'projekt_2_zad_2_mnist', 1

def cifar10_to_cnn(device, train):
    cifar           = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_cifar10)
    cifars          = CustomDataset(data=cifar.data, targets=cifar.targets, device=device)
    return cifars, 3, 'projekt_2_zad_2_cifar10', 3
