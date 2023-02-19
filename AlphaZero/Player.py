from AlphaZero.MCTS import MCTS
from numpy.random import choice
from copy import deepcopy

class DeepLearningPlayer:
    def __init__(self, model, rollouts=1600, save_tree=True, use_policy=True):
        self.name = "AlphaHex"
        self.model = model
        self.rollouts = rollouts
        self.MCTS = None
        self.save_tree = save_tree
        self.use_policy = use_policy


    def getMove(self, game):
        init_board = deepcopy(game.board)
        if self.MCTS is None or not self.save_tree:
            self.MCTS = MCTS(self.model,use_policy=self.use_policy)
        if self.save_tree and game in self.MCTS.visited_nodes:
            root_node = self.MCTS.visited_nodes[game]
        else:
            root_node = self.MCTS.expandRoot(game)

        self.MCTS.runSearch(root_node, self.rollouts)
        searchProbabilities = self.MCTS.getSearchProbabilities(root_node)

        prob_items = searchProbabilities.items()

        game.board = init_board

        best_move = max(prob_items, key=lambda c: c[1])
        return best_move[0]


class RandomPlayer:
    def __init__(self):
        self.name = "Random Player"

    def getMove(self, game):
        from random import choice
        return choice(game.availableMoves)
