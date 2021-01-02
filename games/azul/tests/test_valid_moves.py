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


def test_all_4_of_a_kind():
    """
    If all factories have 4 of a kind tiles on them,
    need to make sure game doesn't crash!!
    """
    azs = AzulSimulator(2, random_seed=2)
    azs.load(azs)

    # Set to all 4 of a kind!
    for factory in azs.factories:
        factory_tile_type = None
        for index, tile in enumerate(factory.tiles):
            if factory_tile_type == None:
                factory_tile_type = tile
            elif tile.color != factory_tile_type.color:
                azs.bag.append(tile)
                next_value = next(t for t in azs.bag
                                  if t.color == factory_tile_type.color)
                factory.tiles[index] = next_value
                azs.bag.remove(next_value)
    azs.print_board()

    round_over = False
    while not round_over:
        moves = azs.valid_moves()
        assert len(moves) > 0
        print(f"MOVES {[azs.parse_integer_move(move) for move in moves]}")
        move = moves[0]
        round_over = azs.make_move(move)
