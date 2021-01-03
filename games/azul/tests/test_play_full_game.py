import os
from games.azul.azul_simulator import AzulSimulator, Factory, Tile, FIRST_MOVER_TILE
from games.azul.azul_controller import AzulController

from experiments.azul_experiment import Net

import multiprocessing
from multiprocessing import Manager
import torch
import pytest

HERE = os.path.dirname(__file__)

multiprocessing.set_start_method('spawn')

BEST_MOVE_TEST_CASES = [(
    1,
    0,
), (
    2,
    1,
)]


def test_play_full_game():
    """
    Check undoing moves
    """
    azs = AzulSimulator(2, random_seed=15)
    azs.load(azs)

    while not azs.over():
        moves = azs.valid_moves()
        move = moves[0]
        azs.make_move(move)
        azs.print_board()

    assert azs.score() == 1


def test_playout():
    """
    Test playout
    """
    manager = Manager()
    model = Net()
    model.compile(torch.optim.Adadelta, lr=0.3)

    model.maybe_load_from_file(path=os.path.join(HERE, 'model.pt'))
    controller = AzulController(manager, model)

    azs = AzulSimulator(2, random_seed=15)
    azs.load(azs)
    print(azs.valid_moves())
    azs.print_board()
    # Move red to player 1's first tile row, evaluate how good this is.
    azs.make_move(0)

    score = controller.playout(azs)
    assert score == -1


@pytest.mark.parametrize('moves_left,expected_move_index',
                         BEST_MOVE_TEST_CASES)
def test_best_move(moves_left, expected_move_index):
    """
    Test best move, especially when there is only one move available!
    """
    manager = Manager()
    model = Net()
    model.compile(torch.optim.Adadelta, lr=0.3)

    model.maybe_load_from_file(path=os.path.join(HERE, 'model.pt'))
    controller = AzulController(manager, model)

    azs = AzulSimulator(2, random_seed=15)
    azs.load(azs)
    print(azs.valid_moves())
    azs.print_board()
    # Move red to player 1's first tile row, evaluate how good this is.
    moves = azs.valid_moves()
    while len(moves) > moves_left:
        move = moves[0]
        azs.make_move(move)
        moves = azs.valid_moves()
    azs.print_board()
    move = controller.best_move(azs, playouts=1)
    assert move == moves[expected_move_index]