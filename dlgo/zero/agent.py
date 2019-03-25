import numpy as np
from keras.optimizers import SGD

from ..agent import Agent

class Branch:
    def __init__(self, prior):
        self.prior = prior
        self.visit_count = 0
        self.total_value = 0.0

class ZeroTreeNode:
    def __init__(self, state, value, priors, parent, last_move):
        self.state = state
        self. value = value
        self.parent = parent
        self.last_move = last_move
        self.total_visit_count = 1
        self.branches = {}
        for move, p in priors.items():
            if state.is_valid_move(move):
                self.branches[move] = Branch(p)
        self.children = {}

    def moves(self):
        return self.branches.keys()

    def add_child(self, move, child_node):
        self.children[move] = child_node

    def has_child(self, move):
        return move in self.children

    def record_visit(self, move, value):
        self.total_visit_count += 1
        self.branches[move].visit_count += 1
        self.branches[move].total_value += value

    def expected_value(self, move):
        branch = self.branches[move]
        if branche.visit_count == 0:
            return 0.0
        return branch.total_value / branch.visit_count

    def prior(self, move):
        return self.branches[move].prior

    def visit_count(self, move):
        if move in self.branches:
            return self.branches[move].visit_count
        return 0

class ZeroAgent(Agent):
    def __init__(self, model, encoder, rounds_per_move=1600, c=2.0):
        self._model = model
        self._encoder = encoder
        self._collector = None
        self._num_rounds = rounds_per_move
        self._c = c

    def select_move(self, game_state):
        root = self.create_node(game_state)
        for i in range(self._num_rounds):
            node = root
            next_move = self.select_branch(node)
            while node.has_child(next_move):
                node = node.get_child(next_move)
                next_move = self.select_branch(node)
            new_state = node.state.apply_move(next_move)
            child_node = self.create_node(new_state, parent=node)
            move = next_move
            value = -1 * child_node.value
            while node is not None:
                node.record_visit(move, value)
                move = node.last_move
                node = node.parent
                value = -1 * value
        if self.collector is not None:
            root_state_tensor = self._encoder.encode(game_state)
            visit_counts = np.array([
                root.visit_count(
                    self._encoder.decode_move_index(idx))
                for idx in range(self._encoder.num_moves())
            ])
            self._collector.record_decision(root_state_tensor, visit_counts)
        return max(root.moves(), key=root.visit_count)

    def select_branch(self, node):
        total_n = node.total_visit_count

        def score_branch(move):
            q = node.expected_value(move)
            p = node.prior(move)
            n = node.visit_count(move)
            return q + self._c * p * np.sqrt(total_n) / (n + 1)

        return max(node.moves(), key=score_branch)

    def create_node(self, game_state, move=None, parent=None):
        state_tensor = self._encoder.encode(game_state)
        model_input = np.array([state_tensor])
        priors, values = self._model.predict(model_input)
        priors = priors[0]
        # add Dirichlet noise to the root node
        if parent is None:
            noise = np.random.dirichlet(0.03 * np.ones_like(priors))
            priors = 0.75 * priors + 0.25 * noise
        value = values[0][0]
        move_priors = {
            self._encoder.decode_move_index(idx): p
            for idx, p in enumerate(priors)
        }
        new_node = ZeroTreeNode(game_state, value, move_priors, parent, move)
        if parent is not None:
            parent.add_child(move, new_node)
        return new_node

    def train(self, experience, learning_rate, batch_size):
        num_examples = experience.states.shape[0]
        model_input = experience.states
        visit_sums = np.sum(
            experience.visit_counts, axis=1).reshape((num_examples, 1))

        action_target = experience.visit_counts / visit_sums
        value_target = experience.rewards

        self._model.compile(SGD(lr=learning_rate), loss=['categorical_crossentropy', 'mse'])
        self._model.fit(model_input, [action_target, value_target], batch_size=batch_size)
