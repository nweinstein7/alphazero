from games.azul.azul_simulator import (AzulSimulator, Factory, PlayerBoard,
                                       Tile, PlayerBoard, COLORS,
                                       TILE_WALL_SIZE, tile_number_to_color)
import timeit


class TimeTester(AzulSimulator):
    def initialize_from_obs(self, obs):
        """
        Given an observation matrix, repopulate game board
        """
        self.factories = [Factory([]) for _ in range(0, self.num_factories)]
        self.boards = [PlayerBoard() for _ in range(0, self.num_players)]
        self.bag = []
        self.center = []
        self.box = []
        # Turn is stored after colors in every row, but let's
        # just grab it from the first one.
        self.turn = obs[0][0][len(COLORS) + 1]

        tile_wall_to_fill_in = {
            board: list(range(0, TILE_WALL_SIZE))
            for board in range(0, len(self.boards))
        }
        # Similarly, score is stored after turn.
        # Add score to state:
        for player in range(1, self.num_players + 1):
            self.scores[player] = obs[0, 0, len(COLORS) + 1 + player]

        for tile_number in range(0, self.num_tiles + 1):
            color = tile_number_to_color(tile_number)
            # Iterate over every color, INCLUDING the 1st mover.
            found = False
            for factory in range(0, self.num_factories):
                if obs[tile_number][factory][color] == 1.0:
                    t = Tile(tile_number, color)
                    self.factories[factory].tiles.append(t)
                    found = True
                    break
            if found:
                continue
            if obs[tile_number][self.num_factories][color] == 1.0:
                # Add tile to bag if appropriate.
                t = Tile(tile_number, color)
                self.bag.append(t)
                continue
            if obs[tile_number][self.num_factories + 1][color] == 1.0:
                # Add tile to center if appropriate.
                t = Tile(tile_number, color)
                self.center.append(t)
                continue
            if obs[tile_number][self.num_factories + 2][color] == 1.0:
                # Add tile to box if appropriate.
                t = Tile(tile_number, color)
                self.box.append(t)
                continue

            # populate player boards
            board_index_start = self.num_factories + 3
            for b, board in enumerate(self.boards):
                # populate floor from observation
                if obs[tile_number][b * 31 + board_index_start][color] == 1.0:
                    board.floor.append(Tile(tile_number, color))
                    found = True
                    break
                # populate staging rows
                for y in range(0, 5):
                    # Add 1 for the floor.
                    if obs[tile_number][board_index_start + b * 31 + y +
                                        TILE_WALL_SIZE + 1][color] == 1.0:
                        first_empty_spot = board.staging_rows[y].index(None)
                        tile = Tile(tile_number, color)
                        board.staging_rows[y][first_empty_spot] = tile
                        found = True
                        break

                # populate tile wall
                for x in tile_wall_to_fill_in[b]:
                    if obs[tile_number][board_index_start + b * 31 + x +
                                        1][color] == 1.0:
                        tile = Tile(tile_number, color)
                        board.tile_wall[x] = tile
                        tile_wall_to_fill_in[b].remove(x)
                        print(tile_wall_to_fill_in[b])
                        found = True
                        break
                if found:
                    break


SETUP_1 = """
from games.azul.performance import TimeTester 
tt = TimeTester(2);
tt.load(tt);
state = tt.state();
"""
if __name__ == "__main__":
    print(
        timeit.timeit('tt.initialize_from_obs(state)',
                      setup=SETUP_1,
                      number=1000))
