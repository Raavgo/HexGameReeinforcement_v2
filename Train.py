import torch

from AlphaZero.MCTS import MCTS
import pytorch_lightning as pl
from AlphaZero.AlphaHexLightning import AlphaHexLightning
from AlphaZero.AlphaHex import AlphaHex
from Enviorment.HexGame import HexGame
from AlphaZero.DeeplearningPlayer import DeepLearningPlayer
from torch.utils.data import TensorDataset, DataLoader
import numpy as np


def reshapedSearchProbs(search_probs):
    moves = list(search_probs.keys())
    probs = list(search_probs.values())
    reshaped_probs = np.zeros(64).reshape(8, 8)
    for move, prob in zip(moves, probs):
        reshaped_probs[move[0]][move[1]] = prob
    return reshaped_probs.reshape(64)


from copy import deepcopy


def play_game(game, player1, player2, show=True):
    """Plays a game then returns the final state."""
    new_game_data = []
    while not game.isTerminal:
        if show:
            print(game)
        if game.turn == 1:
            m = player1.getMove(game)
        else:
            m = player2.getMove(game)
        if m not in game.availableMoves:
            raise Exception("invalid move: " + str(m))
        node = player1.MCTS.visited_nodes[game]
        if game.turn == 1:
            search_probs = player1.MCTS.getSearchProbabilities(node)
            board = game.board
        if game.turn == -1:
            search_probs = player2.MCTS.getSearchProbabilities(node)
            board = -game.board.T
        reshaped_search_probs = reshapedSearchProbs(search_probs)
        if game.turn == -1:
            reshaped_search_probs = reshaped_search_probs.reshape((8, 8)).T.reshape(64)

        if np.random.random() > 0.5:
            new_game_data.append((board, reshaped_search_probs, None))
        if np.random.random() > 0.5:
            new_game_data.append((board, reshaped_search_probs, None))
        game = game.makeMove(m)
    if show:
        print(game, "\n")

        if game.winner != 0:
            print("player", game.winner, "(", end='')
            print((player1.name if game.winner == 1 else player2.name) + ") wins")
        else:
            print("it's a draw")
    outcome = 1 if game.winner == 1 else 0
    new_training_data = [(board, searchProbs, outcome) for (board, searchProbs, throwaway) in new_game_data]
    # add training data
    # training_data += new_training_data
    return game, new_training_data


def selfPlay(current_model, numGames, training_data, show=False):
    for i in range(numGames):
        print('Game #: ' + str(i))

        g = HexGame(8)
        player1 = DeepLearningPlayer(current_model, rollouts=400)
        player2 = DeepLearningPlayer(current_model, rollouts=400)

        game, new_training_data = play_game(g, player1, player2, show)
        training_data += new_training_data
    return training_data


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


def trainModel(current_model, training_data, iteration, trainer):
    new_model = current_model

    train_x, train_y = formatTrainingData(training_data)
    np.savez('training_data_' + str(iteration), train_x, train_y['policy_out'], train_y['value_out'])
    loader = DataLoader(
        TensorDataset(torch.Tensor(train_x), torch.Tensor(train_y['policy_out']), torch.Tensor(train_y['value_out'])),
        num_workers=2)
    trainer.fit(new_model, loader, None)
    trainer.save_checkpoint('model_' + str(iteration) + '.ckpt')
    return new_model


def evaluateModel(new_model, current_model, iteration, trainer):
    numEvaluationGames = 40
    newChallengerWins = 0
    threshold = 0.55
    player_new_m = DeepLearningPlayer(new_model, rollouts=400)
    player_old_m = DeepLearningPlayer(current_model, rollouts=400)

    # play 40 games between best and latest models
    for i in range(int(numEvaluationGames // 2)):
        g = HexGame(8)
        game, _ = play_game(g, player_new_m, player_old_m, show=False)

        if game.winner:
            newChallengerWins += game.winner

    for i in range(int(numEvaluationGames // 2)):
        g = HexGame(8)
        game, _ = play_game(g, player_old_m, player_new_m, show=False)

        if game.winner == -1:
            newChallengerWins += abs(game.winner)

    winRate = newChallengerWins / numEvaluationGames

    print('evaluation winrate ' + str(winRate))
    text_file = open("evaluation_results.txt", "w")
    text_file.write("Evaluation results for iteration" + str(iteration) + ": " + str(winRate) + '\n')
    text_file.close()

    if winRate >= threshold:
        trainer.save_checkpoint('model_best.ckpt')
        return new_model
    return current_model


def train(iterations=10, current_model=None):
    if current_model is None:
        current_model = AlphaHex(board_size=8)

    for i in range(iterations):
        trainer = pl.Trainer(max_epochs=50, log_every_n_steps=10, accelerator="gpu", devices=1)
        training_data = []

        training_data = selfPlay(current_model, 200, training_data)
        np.save('training_data_raw_0', training_data)
        new_model = trainModel(current_model, training_data, i, trainer)
        current_model = evaluateModel(new_model, current_model, i, trainer)


if __name__ == "__main__":
    import time

    start = time.time()
    training_data = np.load('training_data_raw_0.npy', allow_pickle=True)
    current_model = AlphaHexLightning()

    train(20, current_model)
    end = time.time()
    print(end - start)