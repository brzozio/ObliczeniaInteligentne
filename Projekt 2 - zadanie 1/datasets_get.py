import torch 
from sklearn import datasets
from torch.utils.data import Dataset

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
    iris_dataset  = CustomDataset(data=iris.data, targets=iris.target, device=device)
    return iris_dataset, 4, 3, 'iris', 20

def wine(device):
    wine          = datasets.load_wine()
    wine_dataset  = CustomDataset(data=wine.data, targets=wine.target, device=device)
    return wine_dataset, 13, 3, 'wine', 100

def breast_cancer(device):
    breast_cancer          = datasets.load_breast_cancer()
    breast_cancer_dataset  = CustomDataset(data=breast_cancer.data, targets=breast_cancer.target, device=device)
    return breast_cancer_dataset, 30, 2, 'breast_cancer', 300