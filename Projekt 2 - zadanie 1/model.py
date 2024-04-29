import numpy as np
import pandas as pd
import torch as torch
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

num_epochs = 1000


if __name__ == "__main__":
    model = MLP(input_size=10, hidden_layer_size=20, classes=3)
    criteria = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01) 

    for epoch in range(num_epochs):
        outputs = model(inputs)
        loss = criteria(outputs, labels)
        
        # Zerowanie gradientów, aby uniknąć akumulacji w kolejnych krokach
        optimizer.zero_grad()
        
        # Backpropagation: Obliczenie gradientów
        loss.backward()
        
        # Aktualizacja wag
        optimizer.step()
        
        # Wyświetlenie postępu trenowania
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))