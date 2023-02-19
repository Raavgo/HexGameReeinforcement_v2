# load numpy array from npy file
from typing import List

from numpy import load, save
import numpy as np
import os
from Utility import get_epoch

if __name__ == "__main__":
    path = './numpy_bin'
    files: List[str] = list(filter(lambda x: 'total_train_data_epoch_' not in x, os.listdir(path)))
    if len(files) == 0:
        exit()
    total_train_date = np.concatenate([load(f'{path}/{x}', allow_pickle=True) for x in files])
    epoch = get_epoch(path)
    save(f'{path}/total_train_data_epoch_{epoch}.npy', total_train_date)

    for file in files:
        os.remove(f'{path}/{file}')


