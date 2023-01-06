import numpy as np
from numpy import sqrt


class Node(object):
    """Node used in MCTS"""
    def __init__(self, state, parent_node, prior_prob):
        self.state = state
        self.children = {} # maps moves to Nodes
        self.visits = 0
        self.value = 0.5
        # self.value = 0.5 if parent_node is None else parent_node.value
        self.prior_prob = prior_prob
        self.prior_policy = np.zeros((8, 8))
        self.parent_node = parent_node

    def __repr__(self):
        return f"""{self.parent_node} {self.children}"""

    def updateValue(self, outcome, debug=False):
        """Updates the value estimate for the node's state."""
        if debug:
            print('visits: ', self.visits)
            print('before value: ', self.value)
            print('outcome: ', outcome)
        self.value = (self.visits*self.value + outcome)/(self.visits+1)
        self.visits += 1
        if debug:
            print('updated value:', self.value)
    def UCBWeight_noPolicy(self, parent_visits, UCB_const, player):
        if player == -1:
            return (1-self.value) + UCB_const*sqrt(parent_visits)/(1+self.visits)
        else:
            return self.value + UCB_const*sqrt(parent_visits)/(1+self.visits)
    def UCBWeight(self, parent_visits, UCB_const, player):
        """Weight from the UCB formula used by parent to select a child."""
        if player == -1:
            return (1-self.value) + UCB_const*self.prior_prob/(1+self.visits)
        else:
            return self.value + UCB_const*self.prior_prob/(1+self.visits)