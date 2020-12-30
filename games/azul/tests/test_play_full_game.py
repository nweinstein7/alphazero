from games.azul.azul_simulator import AzulSimulator, Factory, Tile, FIRST_MOVER_TILE


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