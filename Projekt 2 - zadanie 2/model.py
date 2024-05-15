import torch.nn as nn
import torch.nn.functional as F


class CNN_tanh(nn.Module):
    def __init__(self, in_side_len: int, in_channels: int, cnv0_out_channels: int,
                 lin0_out_size: int, lin1_out_size: int,
                 convolution_kernel: int, pooling_kernel: int,
                 reduce_to_dim2: bool = False, cnv1_out_channels: int = 2):

        super(CNN_tanh, self).__init__()

        self.in_side_len: int = in_side_len
        self.in_channels: int = in_channels
        self.cnv0_out_channels: int = cnv0_out_channels
        self.cnv1_out_channels: int = cnv1_out_channels
        self.lin0_out_size: int = lin0_out_size
        self.lin1_out_size: int = lin1_out_size
        self.pooling_kernel: int = pooling_kernel
        self.conv_kernel_size: int = convolution_kernel
        self.reduce_to_dim2: bool = reduce_to_dim2

        self.conv0 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.cnv0_out_channels,
                               kernel_size=self.conv_kernel_size)

        if reduce_to_dim2:
            self.cnv1_out_channels = 2
            self.conv_kernel_size = (self.in_side_len - self.conv_kernel_size + 1) // pooling_kernel
            self.lin0_in_size = 2
        else:
            self.lin0_in_size = (self.in_side_len - self.conv_kernel_size + 1) // pooling_kernel
            self.lin0_in_size = (self.lin0_in_size - self.conv_kernel_size + 1) // pooling_kernel
            self.lin0_in_size = self.lin0_in_size * self.lin0_in_size * cnv1_out_channels

        self.conv1 = nn.Conv2d(in_channels=self.cnv0_out_channels, out_channels=self.cnv1_out_channels,
                               kernel_size=self.conv_kernel_size)

        self.pool = nn.MaxPool2d(kernel_size=self.pooling_kernel)

        self.lin0 = nn.Linear(self.lin0_in_size, self.lin0_out_size)
        self.lin1 = nn.Linear(self.lin0_out_size, self.lin1_out_size)

    def forward(self, out):

        out = self.lin0(out)
        out = F.relu(out)
        out = self.lin1(out)

        return out

    def extract(self, in_data):

        out = self.conv0(in_data)
        out = F.tanh(out)
        out = self.pool(out)

        out = self.conv1(out)
        out = F.tanh(out)

        if not self.reduce_to_dim2:
            out = self.pool(out)

        out = out.view(-1, self.lin0_in_size)

        return out


class CNN_relu(nn.Module):
    def __init__(self, in_side_len: int, in_channels: int, cnv0_out_channels: int,
                 lin0_out_size: int, lin1_out_size: int,
                 convolution_kernel: int, pooling_kernel: int,
                 reduce_to_dim2: bool = False, cnv1_out_channels: int = 2):

        super(CNN_relu, self).__init__()

        self.in_side_len: int = in_side_len
        self.in_channels: int = in_channels
        self.cnv0_out_channels: int = cnv0_out_channels
        self.cnv1_out_channels: int = cnv1_out_channels
        self.lin0_out_size: int = lin0_out_size
        self.lin1_out_size: int = lin1_out_size
        self.pooling_kernel: int = pooling_kernel
        self.conv_kernel_size: int = convolution_kernel
        self.reduce_to_dim2: bool = reduce_to_dim2

        self.conv0 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.cnv0_out_channels,
                               kernel_size=self.conv_kernel_size)

        if reduce_to_dim2:
            self.cnv1_out_channels = 2
            self.conv_kernel_size = (self.in_side_len - self.conv_kernel_size + 1) // pooling_kernel
            self.lin0_in_size = 2
        else:
            self.lin0_in_size = (self.in_side_len - self.conv_kernel_size + 1) // pooling_kernel
            self.lin0_in_size = (self.lin0_in_size - self.conv_kernel_size + 1) // pooling_kernel
            self.lin0_in_size = self.lin0_in_size * self.lin0_in_size * cnv1_out_channels

        self.conv1 = nn.Conv2d(in_channels=self.cnv0_out_channels, out_channels=self.cnv1_out_channels,
                               kernel_size=self.conv_kernel_size)

        self.pool = nn.MaxPool2d(kernel_size=self.pooling_kernel)

        self.lin0 = nn.Linear(self.lin0_in_size, self.lin0_out_size)
        self.lin1 = nn.Linear(self.lin0_out_size, self.lin1_out_size)

    def forward(self, out):

        out = self.lin0(out)
        out = F.relu(out)
        out = self.lin1(out)

        return out

    def extract(self, in_data):

        out = self.conv0(in_data)
        out = F.relu(out)
        out = self.pool(out)

        out = self.conv1(out)
        out = F.relu(out)

        if not self.reduce_to_dim2:
            out = self.pool(out)

        out = out.view(-1, self.lin0_in_size)

        return out


class CNN_sigmoid(nn.Module):
    def __init__(self, in_side_len: int, in_channels: int, cnv0_out_channels: int,
                 lin0_out_size: int, lin1_out_size: int,
                 convolution_kernel: int, pooling_kernel: int,
                 reduce_to_dim2: bool = False, cnv1_out_channels: int = 2):

        super(CNN_sigmoid, self).__init__()

        self.in_side_len: int = in_side_len
        self.in_channels: int = in_channels
        self.cnv0_out_channels: int = cnv0_out_channels
        self.cnv1_out_channels: int = cnv1_out_channels
        self.lin0_out_size: int = lin0_out_size
        self.lin1_out_size: int = lin1_out_size
        self.pooling_kernel: int = pooling_kernel
        self.conv_kernel_size: int = convolution_kernel
        self.reduce_to_dim2: bool = reduce_to_dim2

        self.conv0 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.cnv0_out_channels,
                               kernel_size=self.conv_kernel_size)

        if reduce_to_dim2:
            self.cnv1_out_channels = 2
            self.conv_kernel_size = (self.in_side_len - self.conv_kernel_size + 1) // pooling_kernel
            self.lin0_in_size = 2
        else:
            self.lin0_in_size = (self.in_side_len - self.conv_kernel_size + 1) // pooling_kernel
            self.lin0_in_size = (self.lin0_in_size - self.conv_kernel_size + 1) // pooling_kernel
            self.lin0_in_size = self.lin0_in_size * self.lin0_in_size * cnv1_out_channels

        self.conv1 = nn.Conv2d(in_channels=self.cnv0_out_channels, out_channels=self.cnv1_out_channels,
                               kernel_size=self.conv_kernel_size)

        self.pool = nn.MaxPool2d(kernel_size=self.pooling_kernel)

        self.lin0 = nn.Linear(self.lin0_in_size, self.lin0_out_size)
        self.lin1 = nn.Linear(self.lin0_out_size, self.lin1_out_size)

    def forward(self, out):

        out = self.lin0(out)
        out = F.relu(out)
        out = self.lin1(out)

        return out

    def extract(self, in_data):

        out = self.conv0(in_data)
        out = F.sigmoid(out)
        out = self.pool(out)

        out = self.conv1(out)
        out = F.sigmoid(out)

        if not self.reduce_to_dim2:
            out = self.pool(out)

        out = out.view(-1, self.lin0_in_size)

        return out


class CNN_leaky_relu(nn.Module):
    def __init__(self, in_side_len: int, in_channels: int, cnv0_out_channels: int,
                 lin0_out_size: int, lin1_out_size: int,
                 convolution_kernel: int, pooling_kernel: int,
                 reduce_to_dim2: bool = False, cnv1_out_channels: int = 2):

        super(CNN_leaky_relu, self).__init__()

        self.in_side_len: int = in_side_len
        self.in_channels: int = in_channels
        self.cnv0_out_channels: int = cnv0_out_channels
        self.cnv1_out_channels: int = cnv1_out_channels
        self.lin0_out_size: int = lin0_out_size
        self.lin1_out_size: int = lin1_out_size
        self.pooling_kernel: int = pooling_kernel
        self.conv_kernel_size: int = convolution_kernel
        self.reduce_to_dim2: bool = reduce_to_dim2

        self.conv0 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.cnv0_out_channels,
                               kernel_size=self.conv_kernel_size)

        if reduce_to_dim2:
            self.cnv1_out_channels = 2
            self.conv_kernel_size = (self.in_side_len - self.conv_kernel_size + 1) // pooling_kernel
            self.lin0_in_size = 2
        else:
            self.lin0_in_size = (self.in_side_len - self.conv_kernel_size + 1) // pooling_kernel
            self.lin0_in_size = (self.lin0_in_size - self.conv_kernel_size + 1) // pooling_kernel
            self.lin0_in_size = self.lin0_in_size * self.lin0_in_size * cnv1_out_channels

        self.conv1 = nn.Conv2d(in_channels=self.cnv0_out_channels, out_channels=self.cnv1_out_channels,
                               kernel_size=self.conv_kernel_size)

        self.pool = nn.MaxPool2d(kernel_size=self.pooling_kernel)

        self.lin0 = nn.Linear(self.lin0_in_size, self.lin0_out_size)
        self.lin1 = nn.Linear(self.lin0_out_size, self.lin1_out_size)

    def forward(self, out):

        out = self.lin0(out)
        out = F.relu(out)
        out = self.lin1(out)

        return out

    def extract(self, in_data):

        out = self.conv0(in_data)
        out = F.leaky_relu(out, negative_slope=0.33)
        out = self.pool(out)

        out = self.conv1(out)
        out = F.leaky_relu(out, negative_slope=0.33)

        if not self.reduce_to_dim2:
            out = self.pool(out)

        out = out.view(-1, self.lin0_in_size)

        return out

