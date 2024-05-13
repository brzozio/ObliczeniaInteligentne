import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, in_side_len: int, in_channels: int, cnv0_out_channels: int,
                 cnv1_out_channels: int, lin0_out_size: int, lin1_out_size: int, pooling_kernel: int):

        super(CNN, self).__init__()

        self.in_side_len: int = in_side_len
        self.in_channels: int = in_channels
        self.cnv0_out_channels: int = cnv0_out_channels
        self.cnv1_out_channels: int = cnv1_out_channels
        self.lin0_out_size: int = lin0_out_size
        self.lin1_out_size: int = lin1_out_size
        self.pooling_kernel: int = pooling_kernel

        conv_kernel_size: int = 3

        self.lin0_in_size = (self.in_side_len - conv_kernel_size + 1) // pooling_kernel
        self.lin0_in_size = (self.lin0_in_size - conv_kernel_size + 1) // pooling_kernel
        self.lin0_in_size = self.lin0_in_size * self.lin0_in_size * cnv1_out_channels

        self.conv0 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.cnv0_out_channels,
                               kernel_size=conv_kernel_size)
        self.conv1 = nn.Conv2d(in_channels=self.cnv0_out_channels, out_channels=self.cnv1_out_channels,
                               kernel_size=conv_kernel_size)

        self.pool = nn.MaxPool2d(kernel_size=self.pooling_kernel)

        self.lin0 = nn.Linear(self.lin0_in_size, self.lin0_out_size)
        self.lin1 = nn.Linear(self.lin0_out_size, self.lin1_out_size)

    def forward(self, out):

        out = self.conv0(out)
        out = F.relu( out)
        out = self.pool(out)

        out = self.conv1(out)
        out = F.relu(out)
        out = self.pool(out)

        out = out.view(-1, self.lin0_in_size)

        out = self.lin0(out)
        out = F.relu(out)

        out = self.lin1(out)

        return out
