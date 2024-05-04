import transforms
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import classes_Model as cls
import matplotlib.pyplot as plt
import itertools

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    """
    data = cls.DataMnist(batch_size=64)
    #cls.precompute_masks(data.training_data)

    model: cls.LinearModel = cls.LinearModel(dim_data=28*28, dim_target=10, dim_mid_layer=10)
    """

    data = np.genfromtxt("masks_snake.csv", delimiter=";")
    figure = plt.figure(figsize=(28, 28))
    dejta = np.zeros((28,28))
    for y in range(28):
        for x in range(28):
            dejta[y][x] = data[9][28*y+x]
    plt.imshow(dejta,cmap='grey')
    plt.show()






