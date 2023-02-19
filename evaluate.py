from AlphaZero.Player import DeepLearningPlayer
from AlphaZero.AlphaHexLightning import AlphaHexLightning
from Enviorment.HexGame import HexGame
from Utility import getArgs
import os
import shutil

def play_game(game, player_1, player_2, show, flip=False):
    if flip:
        player_dict = {1: player_2, -1: player_2}
    else:
        player_dict = {1: player_1, -1: player_1}

    while not game.isTerminal:
        if show:
            print(game)
        turn = game.turn
        player = player_dict[turn]
        m = player.getMove(game)

        if m not in game.availableMoves:
            raise Exception("invalid move: " + str(m))

        game = game.makeMove(m)
    winner = player_1 if game.winner == 1 else player_2
    if show:
        print(game, "\n")
        if game.winner != 0:
            print(f'Player {winner.name} wins')

    return winner


def evaluateModel(current_model, best_model, iteration):
    #Make sure we always flip the board
    iteration = iteration + 1 if iteration%2==1 else iteration
    index = int(os.environ['SLURM_ARRAY_TASK_ID'])

    player_current = DeepLearningPlayer(current_model, rollouts=400)
    player_current.name = "current_model"

    player_best = DeepLearningPlayer(best_model, rollouts=400)
    player_best.name = "best_model"

    wins = {player_current: 0, player_best: 0}
    flip = False


    for i in range(iteration):
        game = HexGame(8)
        wins[play_game(game, player_current, player_best, show=False, flip=flip)] += 1
        flip = not flip

    text_file = open(f"eval/evaluation_results_{index}.txt", "w")
    text_file.write(str(wins[player_current]/iteration)+"\n")
    text_file.write(str(wins[player_current]) + "\n")
    text_file.write(str(wins[player_best]) + "\n")


if __name__ == "__main__":
    args = getArgs()
    current_model = args["model_path"] + '/current_model.ckpt'
    best_model = args["model_path"]+"/best_model.ckpt"

    if not os.path.isfile(best_model):
        shutil.copy(current_model, best_model)
        exit()
    os.environ['SLURM_ARRAY_TASK_ID'] = "1234"
    current_model = AlphaHexLightning.load_from_checkpoint(current_model, n=8)
    best_model = AlphaHexLightning.load_from_checkpoint(best_model, n=8)
    evaluateModel(current_model, best_model, 2)

