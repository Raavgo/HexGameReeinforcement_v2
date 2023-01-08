import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F

from AlphaZero.ValueHeadLayer import ValueHeadLayer
from AlphaZero.PolicyHeadLayer import PolicyHeadLayer
from AlphaZero.ResidualLayer import ResidualLayer
from AlphaZero.Convolution import build_convolution
args = dict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
    'num_channels': 512,
})

class AlphaHex(nn.Module):
    def __init__(self, filter_size=256, board_size=8, dropout=0.3):
        super(AlphaHex, self).__init__()
        self.board_size = board_size
        self.dropout = dropout

        self.conv_1 = build_convolution(in_channels=1, filter_size=filter_size, kernel=(3, 3), stride=(1, 1), padding=1)
        self.conv_2 = build_convolution(in_channels=filter_size, filter_size=filter_size, kernel=(3, 3), stride=(1, 1), padding=0)

        self.linear_layer = nn.Linear(1024, 512)
        self.linear_layer_2 = nn.Linear(1024, 512)

        self.batch_norm = nn.BatchNorm1d(1024)
        self.batch_norm_2 = nn.BatchNorm1d(512)

        self.policy_head = PolicyHeadLayer(1024, board_size**2)
        self.value_head = ValueHeadLayer(1024)

    def forward(self, x):
        out = x.view(-1, 1, self.board_size, self.board_size)

        out = self.conv_1(out)
        out = self.conv_2(out)
        out = self.conv_2(out)
        out = self.conv_2(out)

        out = out.view(-1)
        self.linear_layer(out)

        policy = self.policy_head(out)
        value = self.value_head(out)
        return policy, value


if __name__ == "__main__":
    from Enviorment.hex_engine_0_5 import hexPosition as HexGame
    g = HexGame(8)
    # Test the AlphaHex model
    print(g.board)
    x = torch.Tensor(g.board)
    #x = torch.randn(1, 1, 8, 8)
    print(x.shape)



    model = AlphaHex(filter_size=256, board_size=8)
    probs, val = model(x)
    print(val.item())
    #print(model(x))

#print(model)
#p, v = model(x)
#print(p,v)
#print(np.argmax(p[0].detach().numpy()))
#print(v)