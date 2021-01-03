import sys, os, random, time, warnings

from itertools import product
import numpy as np
import IPython

from scipy.stats import pearsonr
from utils import hashable, sample
from alphazero import AlphaZeroController

from games.azul.azul_simulator import get_next_move_from_playouts, AzulSimulator


class AzulController(AlphaZeroController):
    def __init__(self, manager, model, T=0.3, C=1.5):
        super().__init__(manager, model, T=T, C=C)

    r"""
    Evaluates the "value" of a state by randomly playing out games starting from that state and noting the win/loss ratio.
    """

    def fit(self, game, target, steps=2):
        dataset = [{'input': game.state(), 'target': np.float(target)}]
        for i in range(0, steps):
            self.model.fit(dataset, batch_size=1, verbose=True)

        return target

    r"""
    Evaluates the "value" of a state using the network estimation function.
    """

    def network_value(self, game):
        dataset = [{'input': game.state(), 'target': None}]
        return self.model.predict(dataset).mean()

    r"""
    Runs a single, random heuristic guided playout starting from a given state. This updates the 'visits' and 'differential'
    counts for that state, as well as likely updating many children states.
    """

    def playout(self, game, expand=150):
        print(f"RUNNING PLAYOUT {expand}")
        if expand == 150:
            print("Sleeping for a bit to allow other processes to run")
            time.sleep(.0001)
        if expand == 0 or game.over():
            score = game.score()
            #print ('X' if game.turn==1 else 'O', score)
            return score

        action_mapping = {}

        for action in game.valid_moves():
            print(f"Making first move with action {action}")
            #game.print_board()
            game.make_move(action)
            print(f"ACTION: {action}")
            action_mapping[action] = self.heuristic_value(game)
            print(f"Heuristic value: {action_mapping[action]}")
            game.undo_move()

        chosen_action = sample(action_mapping, T=self.T)
        prev_turn = game.turn
        game.make_move(chosen_action)
        next_turn = game.turn
        multiplier = -1
        if next_turn == prev_turn:
            multiplier = 1
        print('Playing out branch.')
        score = multiplier * self.playout(game,
                                          expand=expand - 1)  #play branch
        print(f"Done playing branch with score {score}")
        game.undo_move()

        return score

    r"""
        Evaluates the "value" of a state by randomly playing out games starting from that state and noting the win/loss ratio.
        """

    def value(self, game, playouts=100, steps=2, pool=None):
        scores = []
        if pool:
            scores = pool.map(self.playout,
                              [game.copy() for i in range(0, playouts)])
        else:
            # No parallelism.
            scores = [self.playout(game.copy()) for i in range(0, playouts)]
        V = sum(scores) / len(scores)
        dataset = [{'input': game.state(), 'target': V}]
        for i in range(0, steps):
            self.model.fit(dataset, batch_size=1, verbose=True)

        return V
