import numpy as np
import torch.nn as nn
import torch

from AlphaZero.ValueHeadLayer import ValueHeadLayer
from AlphaZero.PolicyHeadLayer import PolicyHeadLayer
from AlphaZero.ResidualLayer import ResidualLayer
from AlphaZero.Convolution import build_convolution

class AlphaHex(nn.Module):
    def __init__(self, in_channels, shape, filter_size=256, depth=40):
        super(AlphaHex, self).__init__()

        input_size = shape[2] - 82
        assert input_size > 1, "Input size must be greater than 1"

        self.conv = build_convolution(in_channels=in_channels, filter_size=filter_size, kernel=(3, 3))
        self.res = [ResidualLayer(filter_size) for _ in range(depth)]
        self.policy_head = PolicyHeadLayer(256, shape=(1,256,input_size,input_size))
        self.value_head = ValueHeadLayer(256, shape=(1,256,input_size,input_size))

    def forward(self, x):
        out = self.conv(x)
        for layer in self.res:
            out = layer(out)

        policy = self.policy_head(out)
        value = self.value_head(out)
        return policy, value


# Test the AlphaHex model
x = torch.randn(1, 8, 100, 100)
model = AlphaHex(8, filter_size=256, depth=40, shape=x.shape)
#print(model)
p, v = model(x)
print(np.argmax(p[0].detach().numpy()))
print(v)