from AlphaZero.Player import DeepLearningPlayer, RandomPlayer, MCTSPlayer
from AlphaZero.AlphaHexLightning import AlphaHexLightning
from Enviorment.HexGame import HexGame
from evaluate import play_game
from Utility import getArgs
import os

def build_player(path):
    model_path = getArgs()["model_path"]
    model = AlphaHexLightning(n=8).load_from_checkpoint(path, n=8)
    player = DeepLearningPlayer(model)
    player.name = path[:-5].replace(model_path+"/", "")
    return player


if __name__ == "__main__":
    model_path = getArgs()["model_path"]
    best_model = f'{model_path}/best_model.ckpt'
    index = int(os.environ['SLURM_ARRAY_TASK_ID'])
    model_paths = os.listdir(model_path)

    model = AlphaHexLightning(n=8)
    player = DeepLearningPlayer(model)
    player.name = "Init Model"
    best_player = build_player(best_model)
   
    flip = False if index%2==0 else True
    game = HexGame(8)
    result = play_game(game=game, player_1=best_player, player_2=player, flip=False, show=False)

    with open(f'eval/init/results_init_{index}.txt', "w") as f:
        f.write(f"{result.name}")


