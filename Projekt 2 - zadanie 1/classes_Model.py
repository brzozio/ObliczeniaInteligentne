import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from datetime import datetime


class LinearModel(torch.nn.Module):

    def __init__(self, dim_data: int, dim_target: int, dim_mid_layer: int) -> None:
        super(LinearModel, self).__init__()
        self.linear1 = torch.nn.Linear(dim_data, dim_mid_layer)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(dim_mid_layer, dim_target)
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x


class DataMnist:
    def __init__(self, batch_size: int):
        # Download training data from open datasets.
        self.training_data = datasets.MNIST(
            root="data",
            train=True,
            download=True,
            transform=ToTensor(),
        )
        # Download test data from open datasets.
        self.test_data = datasets.MNIST(
            root="data",
            train=False,
            download=True,
            transform=ToTensor(),
        )


def precompute_masks(data) -> None:
    mean_digits = np.zeros((10,28*28))
    run = 255
    for data_point in range(len(data)):
        if data_point > run:
            print(run, "done at", datetime.now().strftime("%H:%M:%S"))
            run += 256
        for y in range(28):
            y28 = y*28
            for x in range(28):
                mean_digits[data[data_point][1]][x+y28] += data[data_point][0][0][y][x]

    np.savetxt("masks_snake.csv", mean_digits, delimiter=";")