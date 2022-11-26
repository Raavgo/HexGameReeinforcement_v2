import torch
import torch.nn as nn


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

    def __init__(self, in_channels, shape=(8, 8, 16, 16), boarf_size=8):
        super(PolicyHeadLayer, self).__init__()
        flat_size = (shape[1] * shape[2] * shape[3]) // (shape[1] * 0.5)
        flat_size = int(flat_size)
        self.conv = nn.Conv2d(in_channels, 2, (1, 1))
        self.batch_norm = nn.BatchNorm2d(2)
        self.fc = nn.Linear(flat_size, boarf_size * boarf_size + 1)
        self.fc_2 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.conv(x)
        out = self.batch_norm(out)
        out = self.relu(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.softmax(out)

        return out


# Test the policy head layer
x = torch.randn(1, 8, 16, 16)
model = PolicyHeadLayer(8, shape=x.shape)
print(model)
output = model(x)
print(output.shape)
print(output)
