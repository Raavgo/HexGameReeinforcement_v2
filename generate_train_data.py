from Enviorment.HexGame import HexGame
from AlphaZero.AlphaHexLightning import AlphaHexLightning
from AlphaZero.Player import DeepLearningPlayer

import os
from os import path
from numpy import asarray
from numpy import save
from Utility import get_epoch, reshapedSearchProbs

def play_game(game, player1, player2, show:bool=True, flip:bool=False):
    """Plays a game then returns the final state."""
    new_game_data = []
    if flip:
        player_dict = {1: player2, -1: player1}
    else:
        player_dict = {1: player1, -1: player2}
    while not game.isTerminal:
        if show:
            print(game)
        turn = game.turn
        player = player_dict[turn]
        m = player.getMove(game)

        if m not in game.availableMoves:
            raise Exception("invalid move: " + str(m))

        node = player.MCTS.visited_nodes[game]
        board = game.board if turn == 1 else -game.board.T
        search_probs = player.MCTS.getSearchProbabilities(node)
        reshaped_search_probs = reshapedSearchProbs(search_probs)
        if turn == -1:
            reshaped_search_probs = reshaped_search_probs.reshape((8, 8)).T.reshape(64)

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


    return game, new_training_data


def selfPlay(current_model, num_games=2, training_data=[], show=False, flip=False):
    num_games = num_games+1 if num_games % 2 == 1 else num_games

    for i in range(num_games):
        print('Game #: ' + str(i))

        g = HexGame(8)
        player1 = DeepLearningPlayer(current_model, rollouts=1000)
        player2 = DeepLearningPlayer(current_model, rollouts=1000)

        game, new_training_data = play_game(g, player1, player2, show, flip)
        training_data += new_training_data
        flip = not flip
    return training_data

if __name__ == "__main__":
    model_path = './model_checkpoint'
    data_path = './numpy_bin'

    # create directory if there is none
    os.makedirs(name=model_path, exist_ok=True)
    os.makedirs(name=data_path, exist_ok=True)

    n = 2
    index = int(os.environ['SLURM_ARRAY_TASK_ID'])

    epoch = get_epoch(data_path)

    model_path += "/best.pth"

    model = AlphaHexLightning(8)
    if path.exists(model_path):
        model.load_from_checkpoint(checkpoint_path=model_path)

    train_data = asarray(selfPlay(model, num_games=n), dtype=object)
    save(f"{data_path}/train_data_epoch_{epoch}_{index}.npy", train_data)
