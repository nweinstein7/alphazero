from games.azul.azul_simulator import AzulSimulator, Factory, Tile, FIRST_MOVER_TILE


def test_play_full_game():
    """
    Check undoing moves
    """
    azs = AzulSimulator(2)
    azs.load(azs, random_seed=4)

    while not azs.over():
        moves = azs.valid_moves()
        move = moves[0]
        azs.make_move(move)

    assert azs.score() == 1