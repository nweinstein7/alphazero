from enum import Enum
import random
import numpy as np
import operator
from games.games import AbstractGame
import time
from multiprocessing import Pool, Manager

COLORS = ['RED', 'BLU', 'TRQ', 'YLL', 'BLK']

FIRST_MOVER_TILE = 5

# in our observation array, empty can't be 0 because 0 is RED
EMPTY_TILE_POSITION = 6

# map of which colors go where in tile wall
TILE_WALL_MAP = [[1, 3, 0, 4, 2], [2, 1, 3, 0, 4], [4, 2, 1, 3, 0],
                 [0, 4, 2, 1, 3], [3, 0, 4, 2, 1]]

# there are 25 tiles in the tile wall
TILE_WALL_SIZE = 25

# map of point loss for each floor position
FLOOR_MAP = [-1, -1, -2, -2, -2, -3, -3]

# There are 6 places to place a tile: 5 staging rows
# plus floor.
PLACEMENT_OPTIONS = 6


def sorted_tile_list(tile_list):
    """
    Sort a list of tiles
    """
    return sorted(tile_list, key=lambda tile: tile.number)


class Tile(object):
    def __init__(self, number, color):
        self.number = number
        self.color = color

    def __str__(self):
        if self.color < len(COLORS):
            return COLORS[self.color]
        elif self.color == FIRST_MOVER_TILE:
            return '1st'
        else:
            return ''

    def __eq__(self, t):
        """
        Check equality
        """
        if not t:
            return False
        return self.number == t.number and self.color == t.color


class Factory(object):
    def __init__(self, tiles):
        self.tiles = tiles

    def fetch(self, color):
        """
        Retrieve the colored tiles from this factory

        Returns:
            (fetched_tiles, discard_tiles) - fetched tiles are the tiles that match
            the color; discard tiles are the tiles that need to go in the center
        """
        fetched_tiles = []
        discard_tiles = []
        if any(t.color == color for t in self.tiles):
            # only perform the operation if the move is valid
            print("Valid move.")
            for t in self.tiles.copy():
                # copy the tiles so we can remove as we iterate
                if t.color == color:
                    fetched_tiles.append(t)
                else:
                    discard_tiles.append(t)
                self.tiles.remove(t)
        else:
            print("Invalid move.")
        return fetched_tiles, discard_tiles


class PlayerBoard(object):
    """
    An individual player's board
    """
    def __init__(self):
        self.staging_rows = [[None] * (x + 1) for x in range(0, 5)]
        self.tile_wall = [None] * TILE_WALL_SIZE
        self.floor = []

    def get_tile_wall_row(self, index):
        """
        Given a row index, retrieve the row of tiles
        at that index
        """
        row = []
        for i in range(5):
            row.append(self.tile_wall[index * 5 + i])
        return row


class AzulSimulator(AbstractGame):
    """
    Simulate a game of Azul
    """
    def __init__(self, num_players=2, num_tiles=100):
        self.num_players = num_players
        self.num_tiles = num_tiles
        self.num_factories = self.num_players * 2 + 1
        self.bag = []
        self.factories = []
        self.center = []
        self.boards = []
        self.box = []
        # 1-indexed number indicating which player's turn it is
        # Implies player 1 plays first move
        self.turn = 1
        # Dictionary of scores
        self.scores = {i: 0 for i in range(1, num_players + 1)}
        # Stack of previous game states, for use with undo.
        self.state_stack = []

    def initialize_factories(self):
        """
        Initialize each factory from the bag
        """
        for factory in self.factories:
            for _ in range(0, 4):
                if len(self.bag) == 0:
                    # repopulate from box
                    self.bag = self.box.copy()
                    self.box = []
                factory.tiles.append(self.bag.pop())

    @classmethod
    def load(cls, self, randomize=True):
        # create a bag of tiles
        self.bag = []
        tile_number = 0
        for c in range(0, len(COLORS)):
            for _ in range(0, 20):
                self.bag.append(Tile(tile_number, c))
                tile_number += 1
        # randomize the bag
        if randomize:
            random.shuffle(self.bag)

        # initialize the factories with tiles
        self.factories = [Factory([]) for _ in range(0, self.num_factories)]
        self.initialize_factories()

        # initialize the center with the first mover tile
        self.center = [Tile(self.num_tiles, FIRST_MOVER_TILE)]

        # initialize player boards
        self.boards = [PlayerBoard() for _ in range(0, self.num_players)]
        print("GAME RESET")
        self.print_board()
        return self

    def parse_integer_move(self, move):
        """
        Given an integer, parse it into selection, color, and placement
        """
        # selection: s, placement: p, color: c
        # s, p, c
        # 0, 0, 0 -> 0
        # 0, 0, 1 -> 1
        # 0, 0, 2 -> 2
        # ...
        # 0, 0, 4 -> 4
        # 0, 1, 0 -> 5
        # 0, 1, 1 -> 6
        # ...
        # 0, 1, 4 -> 9
        color = move % len(COLORS)
        color_placements = ((move - color) / len(COLORS))
        placement = color_placements % PLACEMENT_OPTIONS

        placement_selections = (color_placements -
                                placement) / PLACEMENT_OPTIONS
        selection = placement_selections % (self.num_factories + 1)
        return (int(selection), int(color), int(placement))

    def move_to_integer(self, selection, color, placement):
        """
        Given a selection, color, and placement, return the integer
        representation of this move
        """
        result = color + placement * len(COLORS) + selection * (
            PLACEMENT_OPTIONS * len(COLORS))
        return result

    def update_turn(self):
        """
        Change turns
        """
        self.turn = self.turn % self.num_players + 1

    def make_move(self, move):
        """
        Given a move as an integer, perform that move

        Return True if round over, return False otherwise
        """
        round_over = self.act(*self.parse_integer_move(move))
        self.state_stack.append(self.state())
        self.update_turn()
        return round_over

    def undo_move(self):
        """
        Undo the previous move
        """
        past_state = self.state_stack.pop()
        self.initialize_from_obs(past_state)
        self.update_turn()

    def valid_moves(self):
        """
        Get available valid moves for this turn

        Returns:
            an array of integer moves
        """
        possible_selections = []
        for i, f in enumerate(self.factories):
            # Add factories as possible selections
            if len(f.tiles) > 0:
                possible_selections.append(i)
        print(f"Center: {self.center}")
        if len(
                list(
                    filter(lambda tile: tile.color != FIRST_MOVER_TILE,
                           self.center))):
            # Add center as a possible selection
            print("ADDING CENTER")
            possible_selections.append(len(self.factories))

        print(possible_selections)
        # Determine where on board we can place tiles.
        # Validity depends on whether a tile has already been placed
        # with that color on the tile wall, or whether
        # the staging row already has a color chosen.
        board = self.boards[self.turn - 1]
        color_placements = []
        for r, row in enumerate(board.staging_rows):
            staged_tiles = [t for t in row if t is not None]
            if len(staged_tiles) > 0 and len(staged_tiles) < len(row):
                print(
                    f"There are some blank spaces. Row: {r}. Len staged: {len(staged_tiles)}. Len row: {len(row)}"
                )
                # In this case, there are blank spaces available
                color = staged_tiles[0].color
                color_placements.append((color, r))
            elif len(staged_tiles) == 0:
                print(
                    f"This row is empty. Row: {r}. Len staged: {len(staged_tiles)}. Len row: {len(row)}"
                )
                # In this case, the staging row is blank.
                # Need to check the tile wall row to prevent invalid moves.
                for c, _ in enumerate(COLORS):
                    if not any(t is not None and t.color == c
                               for t in board.get_tile_wall_row(r)):
                        # This color has not been filled in on the tile wall, thus is
                        # valid as a choice.
                        color_placements.append((c, r))
        # Add floor as valid move.
        for c, _ in enumerate(COLORS):
            color_placements.append((c, len(board.staging_rows)))

        # Combine selections and color placements.
        possible_moves = []
        for s in possible_selections:
            if s < len(self.factories):
                # Find moves for each factory.
                colors = set([t.color for t in self.factories[s].tiles])
                for c in colors:
                    possible_moves.extend([(s, _c, _p)
                                           for _c, _p in color_placements
                                           if _c == c])
            elif s == len(self.factories):
                # Find moves for center.
                colors = set([
                    t.color for t in self.center if t.color != FIRST_MOVER_TILE
                ])
                print(f"Center colors: {colors}")
                print(f"Color placements: {color_placements}")
                for c in colors:
                    possible_moves.extend([(s, _c, _p)
                                           for _c, _p in color_placements
                                           if _c == c])
        print(f"Possible moves: {possible_moves}")
        return sorted([self.move_to_integer(*mv) for mv in possible_moves])

    def act(self, selection, color, placement):
        """
        Perform a move for a given player
        selection - the location selected (1 of factories or center)
        color - the color tiles to select
        placement - where to place the tiles (staging row or floor)

        Return True if new round, return False if round not over
        """
        tile_row = placement + 1
        if placement == 5:
            tile_row = "floor"
        factory_selection = selection + 1
        if selection == 5:
            factory_selection = "center"
        print("Running step with factory {}, color {}, tile row {}, player {}".
              format(factory_selection, COLORS[color], tile_row, self.turn))

        tiles = []
        if selection >= self.num_factories:
            # selected center
            if len([t for t in self.center if t.color == color]) > 0:
                print("Valid center move.")
                for t in self.center.copy():
                    # make sure to copy so we don't shorten the list as we go
                    if t.color == color or t.color == FIRST_MOVER_TILE:
                        tiles.append(t)
                        self.center.remove(t)
            else:
                print("Invalid center move.")
        else:
            f = self.factories[selection]
            tiles, discard_tiles = f.fetch(color)
            if len(tiles) > 0:
                print("Fetched: {}".format([t.color for t in tiles]))
                print("Discarded: {}".format([t.color for t in discard_tiles]))

            self.center.extend(discard_tiles)
        board = self.boards[self.turn - 1]
        if placement < len(board.staging_rows):
            # place in staging row
            row = board.staging_rows[placement]
            color_index = TILE_WALL_MAP[placement].index(color)
            tile_wall_index = 5 * placement + color_index
            if all(t == None or t.color == color
                   for t in row) and board.tile_wall[tile_wall_index] == None:
                # if the staging row is empty or matches the color, you can
                # add tiles to it. Also, can't add if the tile wall already has that color.
                for i, cell in enumerate(row):
                    if cell == None:
                        tile = next((_t for _t in tiles if _t.color == color),
                                    None)
                        if tile:
                            print("Tile found: {}".format(COLORS[tile.color]))
                            row[i] = tile
                            tiles.remove(tile)
                        else:
                            print("No tiles found")
        board.floor.extend(tiles)
        self.print_board()
        return self.start_new_round()

    def state(self):
        """
        Render game into observable state (a square)
        5 rows of 8, 1 for each factory. Each factory has 4 tiles
        which can be EMPTY, or one of the colors. So, sort the tiles. For example, RRRR is 1111 base 6.

        6 x 21: each factory + center (6), 21 spots (20 for each tile/color combo, 1 for first mover tile)
        ONE FOR EACH PLAYER:
        6 x 25: each staging row row + floor (6), 25 spots (5 possible spots, 5 possible colors)
        1 x 25: each tile spot, 1 if filled in, 0 otherwise (color is predetermined!!)

        So, one possible encoding is 20 x 25??
        """
        np.zeros((self.num_factories, 20))
        obs = np.full(shape=(self.num_tiles + 1,
                             self.num_factories + 3 + (self.num_players * 31)),
                      fill_value=EMPTY_TILE_POSITION)
        for i, f in enumerate(self.factories):
            for t in f.tiles:
                obs[t.number][i] = float(t.color)
        for bag_tile in self.bag:
            obs[bag_tile.number][self.num_factories] = float(bag_tile.color)
        for center_tile in self.center:
            obs[center_tile.number][self.num_factories + 1] = float(
                center_tile.color)
        for box_tile in self.box:
            obs[box_tile.number][self.num_factories + 2] = float(
                box_tile.color)
        # board index start is the first slot for indexing
        # tiles on the board
        board_index_start = self.num_factories + 3
        for b, board in enumerate(self.boards):
            for t in board.floor:
                # 0th position of every chunk of 31 is the floor
                obs[t.number][b * 31 + board_index_start] = float(t.color)
            for x, t in enumerate(board.tile_wall):
                if t:
                    # positions 1 - 25 are for the tile wall
                    obs[t.number][b * 31 + 1 + x + board_index_start] = float(
                        t.color)
            for y, r in enumerate(board.staging_rows):
                for t in r:
                    # positions 26 - 31 are staging rows
                    if t:
                        obs[t.number][b * 31 + TILE_WALL_SIZE + y +
                                      board_index_start] = float(t.color)
        return obs

    def round_over(self):
        """
        Check if a round is still in progress
        - if there are tiles on factories OR
        - if there are tiles in the center
        """
        return all(len(f.tiles) == 0
                   for f in self.factories) and len(self.center) == 0

    def start_new_round(self):
        """
        Check if we need to start a new round, and if so:
            - tally up points
            - move used tiles to box
            - if bag runs out, move box tiles back to bag
        Return reward
        """
        if self.round_over():
            print("New round.")
            for (player_num, board) in enumerate(self.boards):
                board_score = self.scores[player_num + 1]
                for i, sr in enumerate(board.staging_rows):
                    if all(t != None for t in sr):
                        # if the row is complete
                        for j, tile in enumerate(sr):
                            if j == 0:
                                # add the first tile to board
                                index = TILE_WALL_MAP[i].index(tile.color)
                                tile_wall_location = i * 5 + index
                                print("Tile wall location: {}".format(
                                    tile_wall_location))
                                board.tile_wall[tile_wall_location] = tile
                                #tally up score
                                horizontal_tally = 0
                                horizontal_start = tile_wall_location
                                while horizontal_start > i * 5 and board.tile_wall[
                                        horizontal_start - 1] != None:
                                    horizontal_start -= 1
                                print("Horizontal start: {}".format(
                                    horizontal_start))
                                for horizontal_tally_location in range(
                                        horizontal_start, (i + 1) * 5):
                                    print(
                                        "Horizontal tally location: {}".format(
                                            horizontal_tally_location))
                                    if board.tile_wall[
                                            horizontal_tally_location] != None:
                                        horizontal_tally += 1
                                        print("Horizontal tally: {}".format(
                                            horizontal_tally))
                                    else:
                                        break
                                board_score += horizontal_tally

                                vertical_tally = 0
                                vertical_start = tile_wall_location
                                while vertical_start > i * 5 + index and board.tile_wall[
                                        vertical_start - 5] != None:
                                    vertical_start -= 5
                                print("Vertical start: {}".format(
                                    vertical_start))
                                for vertical_tally_location in range(
                                        vertical_start, TILE_WALL_SIZE, 5):
                                    print("Vertical tally location: {}".format(
                                        vertical_tally_location))
                                    if board.tile_wall[
                                            vertical_tally_location] != None:
                                        vertical_tally += 1
                                        print("Vertical tally: {}".format(
                                            vertical_tally))
                                    else:
                                        break
                                if vertical_tally == 1:
                                    # don't double count the piece if there
                                    # is nothing in vertical direction
                                    vertical_tally = 0
                                board_score += vertical_tally
                                print(
                                    "Total reward after row: {} for player {}".
                                    format(board_score, player_num + 1))
                            else:
                                # add remainder of tiles to box
                                self.box.append(tile)
                            # empty this row
                            sr[j] = None
                    else:
                        print("Incomplete row found.")
                # subtract points for floor and put 1st mover tile back
                for i in range(0, len(board.floor.copy())):
                    if i < len(FLOOR_MAP):
                        # floor values are negative
                        print(
                            "Subtracting for board {} amount {} from total {}."
                            .format(player_num + 1, FLOOR_MAP[i], board_score))
                        board_score += FLOOR_MAP[i]
                    # remove tile from floor and add back to box, or center if 1st mover tile
                    t = board.floor.pop()
                    if t.color == FIRST_MOVER_TILE:
                        self.center.append(t)
                    else:
                        self.box.append(t)

                self.scores[player_num + 1] = board_score
            if self.over():
                # When game is over, apply the end of game rewards.
                for (player_num, board) in enumerate(self.boards):
                    board_score = self.scores[player_num + 1]
                    # Complete rows = +2
                    for i in range(0, 5):
                        tiles_in_row = 0
                        for j in range(0, 5):
                            if board.tile_wall[i * 5 + j] != None:
                                tiles_in_row += 1
                        if tiles_in_row == 5:
                            print("Adding total reward for horizontal row")
                            board_score += 2
                    # Complete column = +7
                    for i in range(0, 5):
                        tiles_in_col = 0
                        for j in range(0, 5):
                            if board.tile_wall[i + j * 5] != None:
                                tiles_in_col += 1
                        if tiles_in_col == 5:
                            print("Adding total reward for vertical column")
                            board_score += 7
                    # Complete color = +10
                    for color_index, color in enumerate(COLORS):
                        color_count = 0
                        for tile in board.tile_wall:
                            if tile and tile.color == color_index:
                                color_count += 1
                        if color_count == 5:
                            print("Adding total reward for color {}".format(
                                color))
                            board_score += 10
                    self.scores[player_num + 1] = board_score
            else:
                # If game not over, repopulate all the factories.
                self.initialize_factories()
            print("Scores: {}".format(self.scores))
            return True
        else:
            print("Round not done yet.")
            return False

    def over(self):
        for board in self.boards:
            for i in range(0, 5):
                filled_in = True
                for j in range(0, 5):
                    filled_in = (board.tile_wall[5 * i + j] != None
                                 and filled_in)
                if filled_in:
                    print("Game over.")
                    return True
        return False

    def score(self):
        """
        For now, only works in 2-player mode:
        Returns -1 if "current" player would lose, 1 if they would win, 0 if game not over
        """
        if not self.over():
            return 0
        winning_player = max(self.scores.items(),
                             key=operator.itemgetter(1))[0]
        if self.turn == winning_player:
            # Turn indicates the player who will go next in this case,
            # so you don't want turn to equal winning player.
            return -1
        else:
            return 1

    def initialize_from_obs(self, obs):
        """
        Given an observation matrix, repopulate game board
        """
        self.factories = [Factory([]) for _ in range(0, self.num_factories)]
        self.boards = [PlayerBoard() for _ in range(0, self.num_players)]

        for i in range(0, self.num_tiles + 1):
            for j in range(0, self.num_factories):
                color = obs[i][j]
                if color != EMPTY_TILE_POSITION:
                    t = Tile(i, color)
                    self.factories[j].tiles.append(t)
            bag_color = obs[i][self.num_factories]
            if bag_color != EMPTY_TILE_POSITION:
                self.bag.append(Tile(i, bag_color))
            center_color = obs[i][self.num_factories + 1]
            if center_color != EMPTY_TILE_POSITION:
                self.center.append(Tile(i, center_color))
            box_color = obs[i][self.num_factories + 2]
            if box_color != EMPTY_TILE_POSITION:
                self.box.append(Tile(i, box_color))

            # populate player boards
            board_index_start = self.num_factories + 3
            for b, board in enumerate(self.boards):
                # populate floor from observation
                floor_color = obs[i][b * 31 + board_index_start]
                if floor_color != EMPTY_TILE_POSITION:
                    board.floor.append(Tile(i, floor_color))

                # populate tile wall
                for x in range(0, TILE_WALL_SIZE):
                    tile_wall_color = obs[i][board_index_start + b * 31 + x +
                                             1]
                    if tile_wall_color != EMPTY_TILE_POSITION:
                        board.tile_wall[x] = Tile(i, tile_wall_color)
                # populate staging rows
                for y in range(0, 5):
                    staging_row_color = obs[i][board_index_start + b * 31 + y +
                                               TILE_WALL_SIZE]
                    if staging_row_color != EMPTY_TILE_POSITION:
                        first_empty_spot = board.staging_rows[y].index(None)
                        board.staging_rows[y][first_empty_spot] = Tile(
                            i, staging_row_color)

    def print_board(self):
        print("SCORES")
        print("{}".format(self.scores))

        print("FACTORIES")
        for i, factory in enumerate(self.factories):
            print("{}: {}".format(
                i + 1, ' '.join(sorted([str(t) for t in factory.tiles]))))
        print("CENTER")
        print("{}".format(' '.join(sorted([str(t) for t in self.center]))))

        print("BOARDS")
        for i, board in enumerate(self.boards):
            print("{}:".format(i + 1))
            for j, sr in enumerate(board.staging_rows):
                tile_wall_row = [
                    board.tile_wall[k] for k in range(j * 5, j * 5 + 5)
                ]
                print("\t{}:{}\t|\t{}".format(
                    j + 1, " ".join([str(t) for t in sr]),
                    " ".join([str(t) for t in tile_wall_row])))
            print("\tFloor:{}".format(" ".join([str(t) for t in board.floor])))

    def __repr__(self):
        output = "FACTORIES"
        for i, factory in enumerate(self.factories):
            output += "\n" + "{}: {}".format(
                i + 1, ' '.join(sorted([str(t) for t in factory.tiles])))
        output += "\n" + "CENTER"
        output += "\n" + "{}".format(' '.join(
            sorted([str(t) for t in self.center])))
        output += "\n" + "BOARDS"
        for i, board in enumerate(self.boards):
            output += "\n" + "{}:".format(i + 1)
            for j, sr in enumerate(board.staging_rows):
                tile_wall_row = [
                    board.tile_wall[k] for k in range(j * 5, j * 5 + 5)
                ]
                output += "\n" + "\t{}:{}\t|\t{}".format(
                    j + 1, " ".join([str(t) for t in sr]), " ".join(
                        [str(t) for t in tile_wall_row]))
            output += "\n" + "\tFloor:{}".format(" ".join(
                [str(t) for t in board.floor]))
        return output

    def __eq__(self, azs):
        """
        Comparison method
        """
        assert len(azs.factories) == len(self.factories)
        for i, factory in enumerate(self.factories):
            sorted_azs_tiles = sorted_tile_list(azs.factories[i].tiles)
            for j, t in enumerate(sorted_tile_list(factory.tiles)):
                assert t == sorted_azs_tiles[j]
        sorted_azs_center = sorted_tile_list(azs.center)
        for k, t in enumerate(sorted_tile_list(self.center)):
            assert t == sorted_azs_center[k]
        assert len(azs.bag) == len(self.bag)
        assert len(azs.box) == len(self.box)
        assert len(azs.boards) == len(self.boards)
        return True


def playout(azs, end_time=None):
    """
    Given an azul simulator, playout the full round and find
    the best move.
    """
    valid_moves = azs.valid_moves()
    # Shuffle the moves so we don't always explore the same moves
    random.shuffle(valid_moves)
    # Dictionary of each player's best moves. 1 for player 1, 2 for player 2, etc.
    best_moves = {key: (None, -100) for key in range(1, azs.num_players + 1)}
    playouts = 0
    if end_time != None and time.time() > end_time:
        print("TIMED OUT. Picking randomly.")
        best_moves[azs.turn] = (random.choice(valid_moves), -100)
        return (best_moves, 0)
    loop_end_time = time.time() + 60
    for move in valid_moves:
        if time.time() > loop_end_time:
            print(
                f"TIMED OUT IN LOOP. Returning best so far: {best_moves} after {playouts} playouts."
            )
            break
        print("Valid move: {}".format(azs.parse_integer_move(move)))
        test_simulator = AzulSimulator(azs.num_players)
        test_simulator.initialize_from_obs(azs.state())
        test_simulator.turn = azs.turn
        round_over = test_simulator.make_move(move)
        if round_over:
            print("ROUND OVER IN PLAYOUT")
            for player in best_moves:
                round_score = test_simulator.scores[player]
                print(f"SCORE FOR PLAYER {player} is {round_score}")
                normalized_round_score = round_score
                for key in test_simulator.scores:
                    if key != player:
                        diff = round_score - test_simulator.scores[key]
                        print(f"DIFF WITH PLAYER {key} is {diff}")
                        normalized_round_score += diff
                max_score = max(normalized_round_score, best_moves[player][1])
                if normalized_round_score == max_score:
                    best_moves[player] = (move, max_score)
            playouts += 1
        else:
            new_best_moves, new_playouts = playout(
                test_simulator,
                end_time if end_time != None else time.time() + 10)
            print(f"NEW BEST MOVES: {new_best_moves}")
            playout_move, playout_score = new_best_moves[azs.turn]
            max_score = max(best_moves[azs.turn][1], playout_score)
            if playout_score == max_score:
                print(
                    f"PLAYOUT SCORE {playout_score} is maximized. MOVE: {azs.parse_integer_move(move)}"
                )
                best_moves[azs.turn] = (move, max_score)
                for player in new_best_moves:
                    if player != azs.turn:
                        best_moves[player] = new_best_moves[player]
            playouts += new_playouts
    return (best_moves, playouts)


if __name__ == '__main__':
    print('Playing azul!')
    n_players = int(input("How many players?"))
    azs = AzulSimulator(n_players)
    azs.load(azs)
    while not azs.over():
        azs.print_board()
        integer_move = -1
        valid_moves = azs.valid_moves()
        while not integer_move in valid_moves:
            best_moves, num_playouts = playout(azs)
            azs.print_board()
            print(
                "Recommended Move: {} with score {} after {} playouts".format(
                    azs.parse_integer_move(best_moves[azs.turn][0]),
                    best_moves[azs.turn][1], num_playouts))

            selection = input(
                'Player {}, which factory do you choose? '.format(azs.turn))
            color = input(
                'Which color do you choose? RED: 0, BLU: 1, TRQ: 2, YLL:3, BLK: 4. '
            )
            placement = input('Which tile row do you choose? ')
            integer_move = azs.move_to_integer(
                int(selection) - 1, int(color),
                int(placement) - 1)
            if not integer_move in valid_moves:
                print("Invalid move. Try again. Valid moves: ", valid_moves)
        azs.make_move(integer_move)
        print("State:")
        print(azs.state())
        print("")
        print("======= State from obs: =======")
        demo_state = AzulSimulator(n_players)
        demo_state.initialize_from_obs(azs.state())
        demo_state.print_board()
        print("===============================")
