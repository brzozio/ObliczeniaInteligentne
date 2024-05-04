import numpy as np
import pandas as pd
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size, hidden_layer_size, classes) -> None:
        super(MLP, self).__init__()

        self.lay_1        = nn.Linear(input_size, hidden_layer_size)
        self.activation   = nn.ReLU()
       #self.activation   = nn.Tanh()
        self.lay_2        = nn.Linear(hidden_layer_size, classes)

    def forward(self, x):
        out = self.lay_1(x)
        out = self.activation(out)
        out = self.lay_2(out)
        return out

if __name__ == "__main__":
    pass
    