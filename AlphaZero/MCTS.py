from AlphaZero.Node import Node
from numpy.random import  choice
import numpy as np
import torch

class MCTS:
    def __init__(self, model, UCB_const=2, use_policy=True, use_value=True):
        self.visited_nodes = {}
        self.model = model
        self.UCB_const = UCB_const
        self.use_policy = use_policy
        self.use_value = use_value

    def runSearch(self, root, n_searches):
        for i in range(n_searches):
            selected_node = root
            available_actions = selected_node.state.getActionSpace()

            while len(available_actions) == len(selected_node.children) and not selected_node.state.winner > 0:
                selected_node = self._select(selected_node, debug=False)

            if not selected_node.state.winner > 0:
                if self.use_policy:
                    if selected_node.state not in self.visited_nodes:
                        selected_node = self._expand(selected_node, debug=False)

                    outcome = selected_node.value if root.state.player == 1 else 1 - selected_node.value
                    self._backpropagate(selected_node, root, outcome, debug=False)
                else:
                    moves = selected_node.state.getActionSpace()
                    np.random.shuffle(moves)
                    for move in moves:
                        if not selected_node.state.makeMove(move) in self.visited_nodes:
                            break
            else:
                self._backpropagate(selected_node, root, selected_node.state.winner, debug=False)

    def create_children(self, parent_node):
        if len(parent_node.state.getActionSpace()) != len(parent_node.children):
            for move in parent_node.state.getActionSpace():
                next_state = parent_node.state.makeMove(move, parent_node.state.player)
                child_node = Node(next_state, parent_node, parent_node.prior_policy[move[0]][move[1]])
                parent_node.children[move] = child_node

    def _select(self, parent_node, debug=False):
        '''returns node with max UCB Weight'''

        children = parent_node.children
        items = children.items()
        if self.use_policy:
            UCB_weights = [(v.UCBWeight(parent_node.visits, self.UCB_const, parent_node.state.turn), v) for k, v in
                           items]
        else:
            UCB_weights = [(v.UCBWeight_noPolicy(parent_node.visits, self.UCB_const, parent_node.state.turn), v) for
                           k, v in items]

        # choose the action with max UCB
        node = max(UCB_weights, key=lambda c: c[0])
        if debug:
            print('weight:', node[0])
            print('move:', node[1].state)
            print('value:', node[1].value)
            print('visits:', node[1].visits)
        return node[1]

    def modelPredict(self, state):
        board = torch.Tensor(state.board)
        if state.player == 1:
            board = (-board).T.reshape((1, 1, 8, 8))
        else:
            board = board.reshape((1, 1, 8, 8))
        probs, value = self.model(board)
        value = value.item()
        probs = probs.reshape((8, 8))
        if state.player == 1:
            probs = probs.T
        return probs, value

    def expandRoot(self, state):
        root_node = Node(state, None, 1)
        if self.use_policy or self.use_value:
            probs, value = self.modelPredict(state)
            root_node.prior_policy = probs
        if not self.use_value:
            value = self._simulate(root_node)
        root_node.value = value
        self.visited_nodes[state] = root_node
        self.create_children(root_node)
        return root_node

    def _expand(self, selected_node, debug=False):
        # policy = [selected_node.prior_policy[move] for move in selected_node.state.availableMoves]
        # move = selected_node.state.availableMoves[policy.index(max(policy))]
        # next_state = selected_node.state.makeMove(move)
        # child_node = Node(next_state, selected_node, selected_node.prior_policy[move])
        if self.use_policy or self.use_value:
            probs, value = self.modelPredict(selected_node.state)
            selected_node.prior_policy = probs
        if not self.use_value:
            # select randomly
            value = self._simulate(selected_node)
        if debug:
            print('expanding node', selected_node.state)
        selected_node.value = value.item()
        self.visited_nodes[selected_node.state] = selected_node
        self.create_children(selected_node)
        return selected_node

    def _simulate(self, next_node):
        # returns outcome of simulated playout
        state = next_node.state
        while not state.isTerminal:
            available_moves = state.getActionSpace()
            index = choice(range(len(available_moves)))
            move = available_moves[index]
            state = state.makeMove(move, state.player)
        return (state.winner + 1) / 2

    def _backpropagate(self, selected_node, root_node, outcome, debug=False):
        current_node = selected_node
        # print(outcome)
        if selected_node.state.winner > 0:
            outcome = selected_node.state.winner
        while current_node != root_node:
            if debug:
                print('selected_node: ', selected_node.state)
                print('outcome: ', outcome)
                print('backpropping')
            current_node.updateValue(outcome, debug=False)
            current_node = current_node.parent_node
            # print(current_node.visits)
        # update root node
        root_node.updateValue(outcome)

    def getSearchProbabilities(self, root_node):
        children = root_node.children
        print(children)<
        items = children.items()
        print(items)

        child_visits = [child.visits for action, child in items]
        sum_visits = sum(child_visits)
        print(child_visits)
        if sum_visits != 0:
            normalized_probs = {action: (child.visits / sum_visits) for action, child in items}
        else:
            normalized_probs = {action: (child.visits / len(child_visits)) for action, child in items}
        return normalized_probs