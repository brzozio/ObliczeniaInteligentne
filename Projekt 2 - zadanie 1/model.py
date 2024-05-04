import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim


class MLP(nn.Module):
    def __init__(self, input_size, hidden_layer_size, classes) -> None:
        super(MLP, self).__init__()

        self.func_1 = nn.Linear(input_size, hidden_layer_size)
        self.relu = nn.ReLU()
        self.func_2 = nn.Linear(hidden_layer_size, classes)

    def forward(self, x):
        out = self.func_1(x)
        out = self.relu(out)
        out = self.func_2(out)
        return out

if __name__ == "__main__":
    pass
    