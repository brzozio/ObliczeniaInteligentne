import numpy as np
import pandas as pd
import torch
import os
import matplotlib.pyplot as plt
from sklearn import datasets as sklearn_datasets
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from torchvision import datasets, transforms
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from joblib import dump, load


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomDataset(Dataset):
    def __init__(self, data, targets, device):
        self.data = torch.tensor(data, dtype=torch.double, device=device)
        self.targets = torch.tensor(targets, dtype=torch.long, device=device)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = {'data': self.data[idx], 'target': self.targets[idx]}
        return sample

transform = transforms.Compose([
        transforms.ToTensor()
    ])

def PCA_DIGIT_VALUE(data, targets):
    flattened_mnist = data.data.flatten(start_dim=1)
    pca = PCA(n_components=2)
    flattened_mnist_pca = pca.fit_transform(flattened_mnist)
    mnists              = CustomDataset(data=StandardScaler().fit_transform(flattened_mnist_pca), targets=targets, device=device)
   
    mnists_data_numpy   = mnists.data.cpu().numpy()
    mnists_data_targets = mnists.targets.cpu().numpy()

    np.savetxt('PCA_50_MNISTS_DATA.txt', mnists_data_numpy)
    np.savetxt('PCA_50_MNISTS_TARGETS.txt', mnists_data_targets)



def TSNE_DIGIT_VALUE(data, targets):
    flattened_mnist = data.data.flatten(start_dim=1)
    tsne = TSNE(n_components=2, random_state=42)
    flattened_mnist_tsne = tsne.fit_transform(flattened_mnist)
    mnists               = CustomDataset(data=StandardScaler().fit_transform(flattened_mnist_tsne), targets=targets, device=device)

    mnists_data_numpy   = mnists.data.cpu().numpy()
    mnists_data_targets = mnists.targets.cpu().numpy()

    np.savetxt('TSNE_50_MNISTS_DATA.txt', mnists_data_numpy)
    np.savetxt('TSNE_50_MNISTS_TARGETS.txt', mnists_data_targets)


if __name__ == "__main__":
    mnist = datasets.MNIST(root='data', train=True, download=True, transform=transform)
    first_50_mnists_data    = mnist.data[1:50]
    first_50_mnists_targets = mnist.targets[1:50]

    TSNE_DIGIT_VALUE(data=first_50_mnists_data, targets=first_50_mnists_targets)
    #PCA_DIGIT_VALUE(data=first_50_mnists_data, targets=first_50_mnists_targets)