import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyHeadLayer(nn.Module):
    """
    Policy Head Layer definition
        2 Convolutional Filters (1x1)
        Batch Normalization
        ReLU Activation
        Fully Connected Layer (shape -> 256)
        ReLU Activation
        Fully Connected Layer (256 -> 1)
        Tanh Activation
    """

    def __init__(self, in_channels, out_channels=49):
        super(PolicyHeadLayer, self).__init__()

        self.fc = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        out = self.fc(x)
        out = F.log_softmax(out, dim=0)

        return out
