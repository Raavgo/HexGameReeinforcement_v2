import torch.nn as nn
import torch
from AlphaZero.Convolution import build_convolution
class ValueHeadLayer(nn.Module):
    """
    Value Head Layer definition
        1 Convolutional Filter (1x1)
        Batch Normalization
        ReLU Activation
        Fully Connected Layer
        Tanh Activation
    """
    def __init__(self, in_channels, shape=(8,8,16,16)):
        super(ValueHeadLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, (1, 1))
        self.fc = nn.Linear(shape[1]*shape[2]*shape[3], 8)
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.tanh(out)
        return out

# Test the value head layer
x = torch.randn(8, 8, 16, 16)
model = ValueHeadLayer(8, shape=x.shape)
print(model)
output = model(x)
print(output.shape)
print(output)