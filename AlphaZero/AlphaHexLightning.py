import pytorch_lightning as pl
import torch
from torch import nn
from AlphaZero.AlphaHex import AlphaHex
from AlphaZero.ValueHeadLayer import ValueHeadLayer
from AlphaZero.PolicyHeadLayer import PolicyHeadLayer
from AlphaZero.ResidualLayer import ResidualLayer
from AlphaZero.Convolution import build_convolution


class AlphaHexLightning(pl.LightningModule):
    def __init__(self, filter_size=256, board_size=8, dropout=0.3):
        super(AlphaHexLightning, self).__init__()
        self.board_size = board_size
        self.dropout = dropout

        self.conv_1 = build_convolution(in_channels=1, filter_size=filter_size, kernel=(3, 3), stride=(1, 1), padding=1)
        self.conv_2 = build_convolution(in_channels=filter_size, filter_size=filter_size, kernel=(3, 3), stride=(1, 1),
                                        padding=0)

        self.linear_layer = nn.Linear(1024, 512)
        self.linear_layer_2 = nn.Linear(1024, 512)

        self.batch_norm = nn.BatchNorm1d(1024)
        self.batch_norm_2 = nn.BatchNorm1d(512)

        self.policy_head = PolicyHeadLayer(1024, board_size ** 2)
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


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, pi, v = batch

        x = x.view(x.size(0), -1)
        out_pi, out_v = self(x)
        l_pi = self.loss_pi(pi, out_pi)
        l_v = self.loss_v(v, out_v)
        loss = l_pi + l_v


        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        y_pred = self(x)
        loss = nn.NLLLoss()(torch.log(y_pred), y)
        self.log('val_loss', loss)

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]
