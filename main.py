import torch
from AlphaZero.AlphaHex import AlphaHex
from Enviorment.hex_engine_0_5 import hexPosition as HexGame
from AlphaZero.DeeplearningPlayer import DeepLearningPlayer


x = torch.randn(1, 1, 8, 8)
#g = HexGame(8)
#print(g.board)
model = AlphaHex(board_size=8)

if __name__ == "__main__":
    g = HexGame(8)
    player1 = DeepLearningPlayer(model, rollouts=400, competitive=True)
    player2 = DeepLearningPlayer(model, rollouts=400)
    print(player1.getMove(g))
    print(g.board)
    #g.makeMove()

    # player2 = DeepLearningPlayer(current_model)
    #game, new_training_data = play_game(g, player1, player2, False)
    #training_data += new_training_data



