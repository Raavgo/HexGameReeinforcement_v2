class Node:
    def __init__(self, prior, turn, state, play, parent=None, level=0, value=0):
        self.state = state
        self.turn = turn
        self.children = {}
        self.level = level
        self.value = value
        self.prior = prior
        self.parent = parent
        self.play = play

    def expand(self, action_probs):
        for action, prob in enumerate(action_probs):
            if prob > 0:
                step, next_state = self.play(self.state, self.turn)
                self.children[action] = Node(prior=prob,
                                             turn=self.turn * -1,
                                             state=next_state,
                                             play=self.play,
                                             parent=self,
                                             level=self.level + 1,
                                             value=step)

    def is_leaf(self):
        return len(self.children) == 0

    def is_root(self):
        return self.parent is None

    def __repr__(self):
        return f"Node({self.level})"

    def __str__(self):
        return f"Node({self.level}, parent={self.parent}, children={self.children}, value={self.value})"


# Test the Node class
import numpy as np

board = np.random.randint(0, 3, (8, 8)) - 1
print(board.shape)
board.reshape(1, 1, 8, 8)
print(board.shape)
def dummy_play(board, turn):
    import random
    from copy import deepcopy

    new_board = deepcopy(board)
    action = random.choice(board.get_legal_moves())
    new_board.board[action] = turn
    return action, new_board

class dummy_board:
    def __init__(self, board):
        self.board = board
        self.turn = 1

    def get_legal_moves(self):
        return list(zip(*np.where(self.board == 0)))

    def __str__(self):
        return str(self.board)


def dummy_model_predict(board):
    value_head = 0.5
    policy_head = np.random.random_sample((len(board.get_legal_moves()),))
    return value_head, policy_head


dummy = dummy_board(board)
root = Node(prior=0, turn=1, state=dummy, play=dummy_play)
value, action_probs = dummy_model_predict(root.state)
root.expand(action_probs=action_probs)
root.children[0].expand(action_probs=action_probs)

print(root)
print(root.children[0])
