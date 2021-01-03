import sys, os, random, time, warnings

from itertools import product
import numpy as np
import IPython

from scipy.stats import pearsonr
from utils import hashable, sample

import multiprocessing
from multiprocessing import Pool, Manager

from mcts import MCTSController


class AlphaZeroController(MCTSController):
    def __init__(self, manager, model, T=0.3, C=1.5):
        super().__init__(manager, T=T, C=C)
        self.model = model

    def score(preds, targets):
        score, _ = pearsonr(preds['target'][:, 0], targets['target'])
        base_score, _ = pearsonr(preds['target'][:, 0],
                                 shuffle(targets['target']))
        return "{0:.4f}/{1:.4f}".format(score, base_score)

    r"""
	Evaluates the "value" of a state using the network + exploration heuristic.
	"""

    def heuristic_value(self, game):
        print("Getting heuristic value")
        #N = self.visits.get("total", 1)
        #print(f"N: {N}")
        #Ni = self.visits.get(hashable(game.state()), 1e-9)
        #print(f"Getting network value. Ni: {Ni}")
        V = self.network_value(game)
        #print(f"Done getting network value. {V}")
        #return V + self.C * (np.log(N) / Ni)
        return V

    r"""
	Evaluates the "value" of a state by randomly playing out games starting from that state and noting the win/loss ratio.
	"""

    def value(self, game, playouts=100, steps=2, pool=None):

        V = super().value(game, playouts=playouts, steps=steps, pool=pool)
        dataset = [{'input': game.state(), 'target': V}]
        for i in range(0, steps):
            self.model.fit(dataset, batch_size=1, verbose=True)

        return V

    r"""
	Evaluates the "value" of a state using the network estimation function.
	"""

    def network_value(self, game):
        dataset = [{'input': game.state(), 'target': None}]
        return self.model.predict(dataset).mean()

    r"""
	Chooses the move that results in the highest value state. Also prints out important diagnostic data
	regarding network valuations.
	"""

    def best_move(self, game, playouts=100, pool=None):
        print("Looking for best move.")
        action_mapping = {}
        previous_mapping = {}
        network_mapping = {}

        for action in game.valid_moves():
            game.make_move(action)
            print("Getting network value.")
            previous_mapping[action] = self.network_value(game)
            print(f"Network value before training {previous_mapping[action]}")
            action_mapping[action] = self.value(game,
                                                playouts=playouts,
                                                pool=pool)
            network_mapping[action] = self.network_value(game)
            print(f"Network value after training: {network_mapping[action]}")
            game.undo_move()

        print({a: "{0:.2f}".format(action_mapping[a]) for a in action_mapping})
        print(
            {a: "{0:.2f}".format(previous_mapping[a])
             for a in action_mapping})
        print(
            {a: "{0:.2f}".format(network_mapping[a])
             for a in action_mapping})

        moves = action_mapping.keys()
        if len(moves) > 1 and len(data):
            data1 = [action_mapping[action] for action in moves]
            data2 = [previous_mapping[action] for action in moves]
            data3 = [network_mapping[action] for action in moves]
            R1, p1 = pearsonr(data1, data2)
            R2, p2 = pearsonr(data1, data3)
            print("Correlation before fitting: {0:.4f} (p={1:.4f})".format(
                R1, p1))
            print("Correlation after fitting: {0:.4f} (p={1:.4f})".format(
                R2, p2))
        else:
            print(f"No correlation calculation. Only 1 possible move! {moves}")
        return max(action_mapping, key=action_mapping.get)
