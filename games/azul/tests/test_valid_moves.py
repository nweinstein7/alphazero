from games.azul.azul_simulator import AzulSimulator, Factory, Tile, FIRST_MOVER_TILE


def test_move_to_integer():
    """
    Check getting valid moves
    """
    azs = AzulSimulator(2, random_seed=4)
    azs.load(azs)

    round_over = False
    while not round_over:
        moves = azs.valid_moves()
        print(f"MOVES {[azs.parse_integer_move(move) for move in moves]}")
        move = moves[0]
        round_over = azs.make_move(move)
