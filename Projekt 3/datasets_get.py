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
import os

repo_name = "nteligentne"
path_script = os.path.dirname(os.path.realpath(__file__))
index = path_script.find(repo_name)
path_data = path_script
if index != -1:
   path_data = path_script[:index + len(repo_name)]
   path_data = path_data + "\\data"

print(path_data)


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


def mnist_to_cnn(device, train):
    mnist           = datasets.MNIST(root=path_data, train=train, download=True, transform=transform_mnist)
    mnists          = CustomDataset(data=mnist.data, targets=mnist.targets, device=device)
    mnists.data     = mnists.data.view(-1,1,28,28)
    return mnists

def cifar10_to_cnn(device, train):
    cifar           = torchvision.datasets.CIFAR10(root=path_data, train=train, download=True, transform=transform_cifar10)
    cifars          = CustomDataset(data=cifar.data, targets=cifar.targets, device=device)
    print(cifars.data.size())
    # cifars.data     = cifars.data.reshape(-1,32,32,3)
    cifars.data     = torch.permute(cifars.data, (0, 3, 1, 2))
    print(cifars.data.size())
    
    return cifars

def mnist_extr_conv(device, train, testtrain): #Conv
    #Getting data from .txt file
    mnist  = datasets.MNIST(root=path_data, train=train, download=True, transform=transforms.ToTensor())
    data = np.genfromtxt(path_data+f"\\mean_digit_convolution_{testtrain}_data.txt", delimiter=";")
    print(f"TESTTRAIN: {testtrain}")
    print(f"MNIST TARGET SIZE: {mnist.targets.size()}")
    #mnists = CustomDataset(data=data, targets=mnist.targets, device=device)
    mnists = CustomDataset(data=data, targets=mnist.targets, device=device)
    return mnists


def mnist_extr_diff(device, train, testtrain): #Diff
    #Getting data from .txt file
    mnist  = datasets.MNIST(root=path_data, train=train, download=True, transform=transforms.ToTensor())
    data = np.genfromtxt(path_data+f"\\differential_{testtrain}_data.txt", delimiter=";")
    mnists = CustomDataset(data=data, targets=mnist.targets, device=device)
    return mnists

def iris(device):
    iris          = sklearn_datasets.load_iris()
    iris_dataset  = CustomDataset(data=StandardScaler().fit_transform(iris.data), targets=iris.target, device=device)
    return iris_dataset


def wine(device):
    wine          = sklearn_datasets.load_wine()
    wine_dataset  = CustomDataset(data=StandardScaler().fit_transform(wine.data), targets=wine.target, device=device)
    return wine_dataset


def breast_cancer(device):
    breast_cancer          = sklearn_datasets.load_breast_cancer()
    breast_cancer_dataset  = CustomDataset(data=StandardScaler().fit_transform(breast_cancer.data), targets=breast_cancer.target, device=device)
    return breast_cancer_dataset

