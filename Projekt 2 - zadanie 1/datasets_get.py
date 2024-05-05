import os
import torch 
import numpy as np
from sklearn import datasets
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from torchvision import datasets, transforms
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from joblib import dump, load


class CustomDataset(Dataset):
    def __init__(self, data, targets, device):
        self.data = torch.tensor(data, dtype=torch.double, device=device)
        self.targets = torch.tensor(targets, dtype=torch.long, device=device)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = {'data': self.data[idx], 'target': self.targets[idx]}
        return sample


def iris(device):
    iris          = datasets.load_iris()
    iris_dataset  = CustomDataset(data=StandardScaler().fit_transform(iris.data), targets=iris.target, device=device)
    return iris_dataset, 4, 3, 'iris', 6


def wine(device):
    wine          = datasets.load_wine()
    wine_dataset  = CustomDataset(data=StandardScaler().fit_transform(wine.data), targets=wine.target, device=device)
    return wine_dataset, 13, 3, 'wine', 8


def breast_cancer(device):
    breast_cancer          = datasets.load_breast_cancer()
    breast_cancer_dataset  = CustomDataset(data=StandardScaler().fit_transform(breast_cancer.data), targets=breast_cancer.target, device=device)
    return breast_cancer_dataset, 30, 2, 'breast_cancer', 15


def mnist_flatten(device, train):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    mnist           = datasets.MNIST(root='data', train=train, download=True, transform=transform)
    flattened_mnist = mnist.data.flatten(start_dim=1)
    mnists          = CustomDataset(data=StandardScaler().fit_transform(flattened_mnist), targets=mnist.targets, device=device)
    return mnists, 28*28, 10, 'mnist_flatten', 120


def mnist_extr_PCA(device, train):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    mnist           = datasets.MNIST(root='data', train=train, download=True, transform=transform)
    flattened_mnist = mnist.data.flatten(start_dim=1)
    #Ekstrakcja na dwie cechy
    pca = PCA(n_components=2)
    flattened_mnist_pca = pca.fit_transform(flattened_mnist)
    mnists              = CustomDataset(data=StandardScaler().fit_transform(flattened_mnist_pca), targets=mnist.targets, device=device)
    return mnists, 2, 10, 'mnist_2_features_PCA', 120



def mnist_extr_TSNE(device, train, testtrain):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    file_path = f'flattened_mnist_tsne_afterTransform_{testtrain}.joblib'
    if os.path.exists(file_path):
        mnist                = datasets.MNIST(root='data', train=train, download=True, transform=transform)
        flattened_mnist_tsne = load(f'flattened_mnist_tsne_afterTransform_{testtrain}.joblib') 
        mnists               = CustomDataset(data=StandardScaler().fit_transform(flattened_mnist_tsne), targets=mnist.targets, device=device)
    else:
        mnist           = datasets.MNIST(root='data', train=train, download=True, transform=transform)
        flattened_mnist = mnist.data.flatten(start_dim=1)
        #Ekstrakcja na dwie cechy
        tsne = TSNE(n_components=2, random_state=42)
        flattened_mnist_tsne = tsne.fit_transform(flattened_mnist)
        dump(flattened_mnist_tsne, f'flattened_mnist_tsne_afterTransform_{testtrain}.joblib') 
        mnists               = CustomDataset(data=StandardScaler().fit_transform(flattened_mnist_tsne), targets=mnist.targets, device=device)

    return mnists, 2, 10, 'mnist_2_features_TSNE', 120


def mnist_extr_3(device, train, testtrain):
    transform = transforms.Compose([
            transforms.ToTensor()
        ])
    #Getting data from .txt file
    mnist  = datasets.MNIST(root='data', train=train, download=True, transform=transform)
    data = np.genfromtxt(f"C:\\Users\\Michał\\Documents\\STUDIA\\II stopień, Informatyka Stosowana - inżynieria oprogramowania i uczenie maszynowe\\I sem\\Obliczenia inteligentne\\Projekt 2 - zadanie 1\\mean_digit_convolution_{testtrain}_data.txt", delimiter=";")
    print(f"TESTTRAIN: {testtrain}")
    print(f"MNIST TARGET SIZE: {mnist.targets.size()}")
    mnists = CustomDataset(data=StandardScaler().fit_transform(data), targets=mnist.targets, device=device)
    return mnists, 10, 10, 'mnist_extr_3', 10


def mnist_extr_4(device, train):
    pass