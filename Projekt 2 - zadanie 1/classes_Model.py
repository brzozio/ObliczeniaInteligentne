import numpy as np
import torch

class model_2Lin_ReLU(torch.nn.Module):

    def __init__(self, layers_size: list[int]) -> None:
        super(model_2Lin_ReLU, self).__init__()

        self.linear1 = torch.nn.Linear(layers_size[0], layers_size[1])
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(layers_size[1], layers_size[2])
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x
