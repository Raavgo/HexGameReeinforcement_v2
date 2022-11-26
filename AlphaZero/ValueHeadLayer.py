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

    def __init__(self, in_channels, shape=(8, 8, 16, 16)):
        super(ValueHeadLayer, self).__init__()
        flat_size = shape[1] * shape[2] * shape[3]
        self.conv = nn.Conv2d(in_channels, in_channels, (1, 1))
        self.batch_norm = nn.BatchNorm2d(in_channels)
        self.fc_1 = nn.Linear(flat_size, 256)
        self.fc_2 = nn.Linear(256, 1)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.batch_norm(out)
        out = self.relu(out)
        out = out.view(out.size(0), -1)
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.fc_2(out)
        out = self.tanh(out)
        return out


# Test the value head layer
x = torch.randn(1, 8, 16, 16)
model = ValueHeadLayer(8, shape=x.shape)
print(model)
output = model(x)
print(output.shape)
print(output)
