import torch.nn as nn
import torch


class ValueHeadLayer(nn.Module):
    """
    Value Head Layer definition
        1 Convolutional Filter (1x1)
        Batch Normalization
        ReLU Activation
        Fully Connected Layer (shape -> 256)
        ReLU Activation
        Fully Connected Layer (256 -> 1)
        Tanh Activation
    """

    def __init__(self, in_channels, out_channels=1):
        super(ValueHeadLayer, self).__init__()

        self.fc = nn.Linear(in_channels, out_channels)
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.fc(x)
        out = self.tanh(out)

        return out


