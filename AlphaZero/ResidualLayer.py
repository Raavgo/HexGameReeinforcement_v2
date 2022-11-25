import torch.nn as nn
import torch
from AlphaZero.Convolution import build_convolution


class ResidualLayer(nn.Module):
    """
    Residual Layer definition
        256 Convolutional Filters (3x3)
        Batch Normalization
        ReLU Activation
        256 Convolutional Filters (3x3)
        Batch Normalization
        Skip Connection
        ReLU Activation
    """

    def __init__(self, in_channels, kernel_size=(2, 2), filter_size=256):
        super(ResidualLayer, self).__init__()
        down_sample_kernel = (kernel_size[0] * 2 - 1, kernel_size[1] * 2 - 1)
        self.conv1 = build_convolution(in_channels, filter_size, kernel_size)
        self.conv2 = build_convolution(filter_size, filter_size, kernel_size, skip_connection=True)
        self.down_sample = nn.Sequential(
            nn.Conv2d(in_channels, filter_size, kernel_size=down_sample_kernel),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out += self.down_sample(x)
        out = self.relu(out)
        return out


# Test the residual layer
x = torch.randn(8, 8, 16, 16)

in_channels = 4
out_channels = 256
kernel_size = (3, 3)
depth = 1

input = torch.randn(16, in_channels, 50, 100)
model = ResidualLayer(in_channels, kernel_size)
print(model)
output = model(input)
print(output.shape)
print(output)
