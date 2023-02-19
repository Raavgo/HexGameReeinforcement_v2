import os
import numpy as np

def get_epoch(data_path):
    return len(
        list(
            filter(
                lambda x: 'total_train_data_epoch' in x, os.listdir(data_path)
            )
        )
    ) + 1

def formatTrainingData(training_data):
    """ training data is an array of tuples (boards, probs, value), we need to reshape into np array of state boards for x, and list of two np arrays of search probs and value for y"""
    x = []
    y_values = []
    y_probs = []
    for (board, probs, value) in training_data:
        x.append(board)
        y_probs.append(probs)
        y_values.append(value)

    # use subset of training data
    train_x = np.array(x).reshape((len(x), 1, 8, 8))
    train_y = {'policy_out': np.array(y_probs).reshape((len(y_probs), 64)), 'value_out': np.array(y_values)}
    return train_x, train_y

def reshapedSearchProbs(search_probs):
    moves = list(search_probs.keys())
    probs = list(search_probs.values())
    reshaped_probs = np.zeros(64).reshape(8, 8)
    for move, prob in zip(moves, probs):
        reshaped_probs[move[0]][move[1]] = prob
    return reshaped_probs.reshape(64)

def getArgs():
    return {
        "data_path": 'numpy_bin',
        "model_path": 'model_checkpoint',
        "eval_path": 'eval'
    }