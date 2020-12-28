import sys, os, random, time, warnings

from itertools import product
import numpy as np
import IPython

from scipy.stats import pearsonr
from utils import hashable, sample
from games.azul.azul_simulator import get_next_move_from_playouts, AzulSimulator


class AzulController():
    def __init__(self, model):
        self.model = model

    r"""
    Evaluates the "value" of a state by randomly playing out games starting from that state and noting the win/loss ratio.
    """

    def fit(self, game, target, steps=2):
        dataset = [{
            'input': game.state()[None, ...],
            'target': np.float(target)
        }]
        for i in range(0, steps):
            self.model.fit(dataset, batch_size=1, verbose=True)

        return target

    r"""
    Evaluates the "value" of a state using the network estimation function.
    """

    def network_value(self, game):
        dataset = [{'input': game.state()[None, ...], 'target': None}]
        return self.model.predict(dataset).mean()

    r"""
    Chooses the move that results in the highest value state. Also prints out important diagnostic data
    regarding network valuations.
    """

    def best_move(self, game, playouts=100):
        print("Looking for best move.")
        action_mapping = {}
        previous_mapping = {}
        network_mapping = {}

        action, target, _playouts = get_next_move_from_playouts(game)

        print(f"Done getting playout value {target}")

        game_copy = AzulSimulator(game.num_players, turn=game.turn)
        game_copy.initialize_from_obs(game.state())
        game_copy.make_move(action)
        previous_mapping[action] = self.network_value(game_copy)
        print(f"Network prediction before fitting {previous_mapping[action]}")
        action_mapping[action] = self.fit(game_copy, target)
        network_mapping[action] = self.network_value(game_copy)
        print(f"Network prediction after fitting: {network_mapping[action]}")

        print({a: "{0:.2f}".format(action_mapping[a]) for a in action_mapping})
        print(
            {a: "{0:.2f}".format(previous_mapping[a])
             for a in action_mapping})
        print(
            {a: "{0:.2f}".format(network_mapping[a])
             for a in action_mapping})

        moves = action_mapping.keys()
        data1 = [action_mapping[action] for action in moves]
        data2 = [previous_mapping[action] for action in moves]
        data3 = [network_mapping[action] for action in moves]
        R1, p1 = pearsonr(data1, data2)
        R2, p2 = pearsonr(data1, data3)
        print("Correlation before fitting: {0:.4f} (p={1:.4f})".format(R1, p1))
        print("Correlation after fitting: {0:.4f} (p={1:.4f})".format(R2, p2))

        return action
