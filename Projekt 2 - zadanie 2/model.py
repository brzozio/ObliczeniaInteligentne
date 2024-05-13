import torch.nn as nn
import torch.nn.functional as F


import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_classes, imsize, channels):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=16, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)

        self.pool = nn.MaxPool2d(kernel_size=2) 
        
        self.fc1 = nn.Linear(32 * ((imsize)//4 - 2) * ((imsize)//4 - 2), 64) 
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, out):
        out = F.relu(self.conv1(out))
        #print(f'OUT SIZE AFTER conv1: {out.size()}')
        out = self.pool(out)
        #print(f'OUT SIZE AFTER POOL 1: {out.size()}')
        out = F.relu(self.conv2(out))
        #print(f'OUT SIZE AFTER conv2: {out.size()}')
        out = self.pool(out)
        #print(f'OUT SIZE AFTER POOL 2: {out.size()}')
        
        out = out.view(out.size(0), -1)
        #print(f'OUT SIZE AFTER FLATTEN: {out.size()}')
        
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out


