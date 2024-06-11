import torch.nn as nn
import torch.nn.functional as F


class CNN_tanh_compose(nn.Module):
    def __init__(self, in_side_len: int, in_channels: int, cnv0_out_channels: int,
                 lin0_out_size: int, lin1_out_size: int,
                 convolution_kernel: int, pooling_kernel: int,
                 reduce_to_dim2: bool = False, cnv1_out_channels: int = 2):

        super(CNN_tanh_compose, self).__init__()

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

        out = self.conv0(out)
        print(f'conv 0 out shape: {out.shape}')
        out = F.tanh(out)
        out = self.pool(out)
        print(f'pool out shape: {out.shape}')

        out = self.conv1(out)
        print(f'conv1 out shape: {out.shape}')
        out = F.tanh(out)

        if not self.reduce_to_dim2:
            out = self.pool(out)

        out = out.view(-1, self.lin0_in_size)


        out = self.lin0(out)
        print(f'lin0 out shape: {out.shape}')
        out = F.relu(out)
        out = self.lin1(out)
        print(f'lin1 out shape: {out.shape}')

        return out


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