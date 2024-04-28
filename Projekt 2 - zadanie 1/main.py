import transforms
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import classes_Model as nn_cls

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Download training data from open datasets.
    training_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # Download test data from open datasets.
    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    batch_size = 64

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    #transforms.precompute_masks()
    model: nn_cls.model_2Lin_ReLU = nn_cls.model_2Lin_ReLU([28*28, 10, 10])



