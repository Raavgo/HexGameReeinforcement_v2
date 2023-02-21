import os.path

import torch

import pytorch_lightning as pl
from AlphaZero.AlphaHexLightning import AlphaHexLightning

from Enviorment.HexGame import HexGame
from AlphaZero.Player import DeepLearningPlayer
from torch.utils.data import TensorDataset, DataLoader

from Utility import formatTrainingData, get_epoch, getArgs
from numpy import load


def train():
    epoch = get_epoch("numpy_bin") - 1
    args = getArgs()

    model_path = args["model_path"]+'/current_model.ckpt'
    if os.path.isfile(model_path):
        model = AlphaHexLightning.load_from_checkpoint(model_path, n=8)
    else:
        model = AlphaHexLightning(n=8)

    trainer = pl.Trainer(max_epochs=100, log_every_n_steps=10, accelerator="gpu")
    training_data = load(f'{args["data_path"]}/total_train_data_epoch_{epoch}.npy', allow_pickle=True)
    train_x, train_y = formatTrainingData(training_data)

    loader = DataLoader(
        TensorDataset(
            torch.Tensor(train_x),
            torch.Tensor(train_y['policy_out']),
            torch.Tensor(train_y['value_out'])
        ),
        num_workers=8
    )

    trainer.fit(model, loader, None)
    trainer.save_checkpoint(f'{args["model_path"]}/model_{epoch}.ckpt')
    trainer.save_checkpoint(f'{args["model_path"]}/current_model.ckpt')
    return trainer.logged_metrics


if __name__ == "__main__":
    import time
    path = './numpy_bin'
    start = time.time()
    metrics = train()
    end = time.time()
    print(end - start)
    
    with open('logs/main_log.out', "a") as f:
        f.write(f"EPOCH: {get_epoch(path)-1}: Current Model trainde 100 epochs with: {metrics['train_loss'].detach().numpy()}\n")
