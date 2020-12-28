from games.azul.azul_simulator import AzulSimulator, Factory, Tile, FIRST_MOVER_TILE


def test_undo_move():
    """
    Check undoing moves
    """
    azs = AzulSimulator(2)
    azs.load(azs, random_seed=4)

    initial_copy = AzulSimulator(2)
    initial_copy.load(initial_copy, random_seed=4)
    round_over = False
    while not round_over:
        moves = azs.valid_moves()
        move = moves[0]
        round_over = azs.make_move(move)

    while len(azs.state_stack) != 0:
        azs.undo_move()
    assert azs == initial_copy
