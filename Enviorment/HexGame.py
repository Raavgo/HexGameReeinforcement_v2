import numpy as np
from scipy.ndimage import label

_adj = np.ones([3, 3], int)
_adj[0, 0] = 0
_adj[2, 2] = 0

RED = u"\033[1;31m"
BLUE = u"\033[1;34m"
RESET = u"\033[0;0m"
CIRCLE = u"\u25CF"

RED_DISK = RED + CIRCLE + RESET
BLUE_DISK = BLUE + CIRCLE + RESET
EMPTY_CELL = u"\u00B7"

RED_BORDER = RED + "-" + RESET
BLUE_BORDER = BLUE + "\\" + RESET


def print_char(i):
    if i > 0:
        return BLUE_DISK
    if i < 0:
        return RED_DISK
    return EMPTY_CELL


class HexGame:

    def __init__(self, size=8):
        self.size = size
        self.turn = 1
        self.board = np.zeros([size, size], int)

        self._moves = None
        self._terminal = None
        self._winner = None
        self._repr = None
        self._hash = None

    def __repr__(self):
        if self._repr is None:
            self._repr = u"\n" + (" " + RED_BORDER) * self.size + "\n"
            for i in range(self.size):
                self._repr += " " * i + BLUE_BORDER + " "
                for j in range(self.size):
                    self._repr += print_char(self.board[i, j]) + " "
                self._repr += BLUE_BORDER + "\n"
            self._repr += " " * (self.size) + " " + (" " + RED_BORDER) * self.size
        return self._repr

    def __hash__(self):
        if self._hash is None:
            self._hash = hash(repr(self))
        return self._hash

    def __eq__(self, other):
        return repr(self) == repr(other)

    def makeMove(self, move):
        hg = HexGame(self.size)
        hg.board = np.array(self.board)
        hg.board[move[0], move[1]] = self.turn
        hg.turn = -self.turn
        return hg

    @property
    def availableMoves(self):
        if self._moves is None:
            self._moves = list(zip(*np.nonzero(np.logical_not(self.board))))
        return self._moves

    @property
    def isTerminal(self):
        if self._terminal is not None:
            return self._terminal
        if self.turn == 1:
            clumps = label(self.board < 0, _adj)[0]
        else:
            clumps = label(self.board.T > 0, _adj)[0]
        spanning_clumps = np.intersect1d(clumps[0], clumps[-1])
        self._terminal = np.count_nonzero(spanning_clumps)
        return self._terminal

    @property
    def winner(self):
        if self.isTerminal:
            return -self.turn
        return 0
