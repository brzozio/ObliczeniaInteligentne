import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, num_classes, imsize, channels):
        super(CNN, self).__init__()
        # Definicja warstw splotowych i poolingowych
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=16, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2) #Zmniejsza dwukrotnie obiekt wejściowy
        
        self.fc1 = nn.Linear(32 * (imsize//4) * (imsize//4), 128) 
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, out, imsize):
        out = F.relu(self.conv1(out))
        out = self.pool(out)
        out = F.relu(self.conv2(out))
        out = self.pool(out)
        
        out = out.view(-1, 32 * (imsize//4) * (imsize//4))
        
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out