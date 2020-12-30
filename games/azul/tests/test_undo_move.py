from games.azul.azul_simulator import AzulSimulator, Factory, Tile, FIRST_MOVER_TILE


def test_undo_move():
    """
    Check undoing moves
    """
    azs = AzulSimulator(2, random_seed=4)
    azs.load(azs)

    initial_copy = AzulSimulator(2, random_seed=4)
    initial_copy.load(initial_copy)

    assert azs == initial_copy
    round_over = False
    while not round_over:
        moves = azs.valid_moves()
        move = moves[0]
        round_over = azs.make_move(move)

    while len(azs.state_stack) != 0:
        azs.undo_move()
    assert azs == initial_copy


def test_undo_move_after_round_over():
    """
    Check undoing moves
    """
    azs = AzulSimulator(2, random_seed=4)
    azs.load(azs)

    round_over = False
    while not round_over:
        moves = azs.valid_moves()
        move = moves[0]
        round_over = azs.make_move(move)
    copy_after_round = azs.copy()
    azs.make_move(2)
    azs.undo_move()
    assert copy_after_round == azs