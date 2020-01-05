from games.azul.azul_simulator import AzulSimulator
import pickle
import os
import pytest

MOVES = [(1, 0, 1), (4, 4, 1), (5, 2, 2)]

HERE = os.path.dirname(__file__)


def test_move_to_integer():
    """
    Check converting moves to integers and back
    """
    azs = AzulSimulator(2)
    for move in MOVES:
        assert azs.parse_integer_move(azs.move_to_integer(*move)) == move


def test_encoding():
    with open(os.path.join(HERE, 'azs.pkl'), 'rb') as pkl:
        azs = pickle.load(pkl)
    for move in MOVES:
        azs.make_move(azs.move_to_integer(*move))
        dup = AzulSimulator(2)
        dup.initialize_from_obs(azs.state())
        assert dup == azs
