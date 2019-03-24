import numpy as np

from keras.optimizers import SGD

from dlgo import encoders
from dlgo import goboard
from dlgo import kerasutil
from dlgo.agent import Agent
from dlgo.agent.helpers import is_point_an_eye

class ValueAgent(Agent):
    def __init__(self, model, encoder, policy='eps-greedy'):
        self._model = model
        self._encoder = encoder
        self._collector = None
        self._temperature = 0.0
        self._policy = policy
        self._last_move_value = 0

    def set_temperature(self, temperature):
        self._temperature = temperature

    def set_collector(self, collector):
        self._collector = collector

    def set_policy(self, policy):
        if policy not in ('eps-greedy', 'weighted'):
            raise ValueError(policy)
        self._policy = policy

    def select_move(self, game_state):
        moves = []
        board_tensors = []
        for move in game_state.legal_moves():
            if not move.is_play:
                continue
            next_state = game_state.apply_move(move)
            board_tensor = self._encoder.encode(next_state)
            moves.append(move)
            board_tensors.append(board_tensor)
        if not moves:
            return goboard.Move.pass_turn()

        board_tensors = np.array(board_tensors)

        # values of the next state from opponent's view
        opp_values = self._model.predict(board_tensors)
        opp_values = opp_values.reshape(len(moves))

        # values from our point of view
        values = 1 - opp_values

        if self._policy == 'eps-greedy':
            ranked_moves = self._rank_moves_eps_greedy(values)
        elif self._policy == 'weighted':
            ranked_moves = self._rank_moves_weighted(values)

        for move_idx in ranked_moves:
            move = moves[move_idx]
            if not is_point_an_eye(game_state.board,
                                   move.point,
                                   game_state.next_player):
                if self._collector is not None:
                    self._collector.record_decision(
                        state=board_tensor,
                        action=self._encoder.encode_point(move.point),
                    )
                self._last_move_value = float(values[move_idx])
                return move
        return goboard.Move.pass_turn()

    def rank_moves_eps_greedy(self, values):
        if np.random.random() < self._temperature:
            values = np.random.random(values.shape)
        # this ranks the moves from worst to best
        ranked_moves = np.argsort(values)
        # return them in best-to-worst order
        return ranked_moves[::-1]

    def rank_moves_weighted(self, values):
        p = values / np.sum(values)
        p = np.power(p, 1.0 / self._temperature)
        p = p / np.sum(p)
        return np.random.choice(np.arange(0, len(values)), size=len(values),
            p=p,
            replace=False)

    def train(self, experience, lr=0.1, batch_size=128):
        opt = SGD(lr=lr)
        self._model.compile(loss='mse', optimizer=opt)

        n = experience.states.shape[0]
        y = np.zeros((n,))
        for i in range(n):
            reward = experience.rewards[i]
            y[i] = 1 if reward > 0 else 0

        self._model.fit(
            experience.states, y,
            batch_size=batch_size,
            epochs=1)

    def serialize(self, h5file):
        h5file.create_group('encoder')
        h5file['encoder'].attrs['name'] = self._encoder.name()
        h5file['encoder'].attrs['board_width'] = self._encoder.board_width
        h5file['encoder'].attrs['board_height'] = self._encoder.board_height
        h5file.create_group('model')
        kerasutil.save_model_to_hdf5_group(self._model, h5file['model'])

    def diagnostics(self):
        return {'value': self._last_move_value}

def load_value_agent(h5file):
    model = kerasutil.load_model_from_hdf5_group(h5file['model'])
    encoder_name = h5file['encoder'].attrs['name']
    if not isinstance(encoder_name, str):
        encoder_name = encoder_name.decode('ascii')
    board_width = h5file['encoder'].attrs['board_width']
    board_height = h5file['encoder'].attrs['board_height']
    encoder = encoders.get_encoder_by_name(
        encoder_name,
        (board_width, board_height))
    return ValueAgent(model, encoder)
