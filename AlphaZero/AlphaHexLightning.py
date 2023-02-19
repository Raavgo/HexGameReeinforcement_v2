import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F


class AlphaHexLightning(pl.LightningModule):
    def __init__(self, n, dropout=0.3):
        super(AlphaHexLightning, self).__init__()
        self.dropout = dropout
        self.board_x, self.board_y = n, n
        self.action_size = n * n

        self.conv1 = nn.Conv2d(1, 512, (3, 3), stride=(1, 1), padding=1)
        self.conv2 = nn.Conv2d(512, 512, (3, 3), stride=(1, 1), padding=1)
        self.conv3 = nn.Conv2d(512, 512, (3, 3), stride=(1, 1))
        self.conv4 = nn.Conv2d(512, 512, (3, 3), stride=(1, 1))

        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(512)
        self.bn3 = nn.BatchNorm2d(512)
        self.bn4 = nn.BatchNorm2d(512)

        self.fc1 = nn.Linear(512 * (self.board_x - 4) * (self.board_y - 4), 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, self.action_size)

        self.fc4 = nn.Linear(512, 1)

    def forward(self, x):
        x = x.view(-1, 1, self.board_x, self.board_y)  # batch_size x 1 x board_x x board_y
        x = F.relu(self.bn1(self.conv1(x)))  # batch_size x num_channels x board_x x board_y
        x = F.relu(self.bn2(self.conv2(x)))  # batch_size x num_channels x board_x x board_y
        x = F.relu(self.bn3(self.conv3(x)))  # batch_size x num_channels x (board_x-2) x (board_y-2)
        x = F.relu(self.bn4(self.conv4(x)))  # batch_size x num_channels x (board_x-4) x (board_y-4)
        x = x.view(-1, 512 * (self.board_x - 4) * (self.board_y - 4))

        x = F.dropout(F.relu((self.fc1(x))), p=self.dropout, training=self.training)  # batch_size x 1024
        x = F.dropout(F.relu((self.fc2(x))), p=self.dropout, training=self.training)  # batch_size x 512

        pi = self.fc3(x)  # batch_size x action_size
        v = self.fc4(x)  # batch_size x 1

        return F.log_softmax(pi, dim=1), torch.tanh(v)

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
