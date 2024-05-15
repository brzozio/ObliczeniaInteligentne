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
from torch.utils.data import DataLoader


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


# Definicja transformacji dla danych testowych
transform_cifar10 = transforms.Compose([
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
    cifar           = torchvision.datasets.CIFAR10(root='./data', train=train, download=True, transform=transform_cifar10)
    cifars          = CustomDataset(data=cifar.data, targets=cifar.targets, device=device)
    print(cifars.data.size())
    # cifars.data     = cifars.data.reshape(-1,32,32,3)
    cifars.data     = torch.permute(cifars.data, (0, 3, 1, 2))
    print(cifars.data.size())
    
    return cifars


def cifar10_to_cnn_AUGMENTED(device, train):
    
    rotate = transforms.Compose([
        # Rotacja obrazu o losowy kąt z zakresu <0,15>
        transforms.RandomRotation(15),          
        transforms.ToTensor(),                  
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  
    ])
    color_jitter = transforms.Compose([
        # Modulacja jasności, kontrastu, nasycenia w obrazie
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  
        transforms.ToTensor(),                                                           
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))       
    ])
    cifar = torchvision.datasets.CIFAR10(root='./data', train=train, download=True)
    print(f'SIZE PRE AUGMENTATION: {len(cifar)}')

    augmented_images = []
    augmented_labels = []
    for data, target in cifar:
        augmented_images.append(rotate(data))
        augmented_labels.append(target)

    cifar = torchvision.datasets.CIFAR10(root='./data', train=train, download=True, transform=transform_cifar10)
    cifar.data = torch.from_numpy(cifar.data)
    cifar.data = torch.permute(cifar.data, (0, 3, 1, 2))
    augmented_images_tensor = torch.stack(augmented_images)
    cifar_extended_data = torch.cat((cifar.data, augmented_images_tensor), dim=0)
    cifar_extended_labels = cifar.targets + augmented_labels

    # Konwertuj listę do postaci tensorów PyTorch i dostosuj kształt danych
    print(f'CIFAR EXT SIZE: {len(cifar_extended_data)}')
    cifars = CustomDataset(data=cifar_extended_data, targets=cifar_extended_labels, device=device)
    cifars.data = torch.permute(cifars.data, (0, 3, 1, 2))
    print(f'SIZE POST AUGMENTATION: {len(cifars)}')

    return cifars



if __name__ == "__main__":
    cifar10_to_cnn_AUGMENTED('cuda', True)