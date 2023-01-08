from AlphaZero.MCTS import MCTS
from numpy.random import choice
from copy import deepcopy

class DeepLearningPlayer:
    def __init__(self, model, rollouts=1600, save_tree=True, competitive=False):
        self.name = "AlphaHex"
        self.model = model
        self.rollouts = rollouts
        self.MCTS = None
        self.save_tree = save_tree
        self.competitive = competitive


    def getMove(self, game):
        init_board = deepcopy(game.board)
        if self.MCTS is None or not self.save_tree:
            self.MCTS = MCTS(self.model)
        if self.save_tree and game in self.MCTS.visited_nodes:
            root_node = self.MCTS.visited_nodes[game]
        else:
            root_node = self.MCTS.expandRoot(game)
        print(root_node)
        self.MCTS.runSearch(root_node, self.rollouts)
        searchProbabilities = self.MCTS.getSearchProbabilities(root_node)
        moves = list(searchProbabilities.keys())
        probs = list(searchProbabilities.values())
        prob_items = searchProbabilities.items()
        print(probs)
        game.board = init_board
        if self.competitive:
            best_move = max(prob_items, key=lambda c: c[1])
            print(best_move)

            return best_move[0]

        else:
            chosen_idx = choice(len(moves), p=probs)
            return moves[chosen_idx]
