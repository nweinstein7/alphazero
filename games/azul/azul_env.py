"""
An environment encoding the game Azul for OpenAI Gym
"""
import numpy as np
import gym
from gym import error, spaces
from azul_simulator import AzulSimulator as AZS

COLORS = ['Red', 'Blue', 'Turquoise', 'Black', 'Yellow']
# 100 tiles
N_TILES = 100


class Spec(object):
    def __init__(self, id, timestep_limit):
        self.id = id
        self.timestep_limit = timestep_limit


class AzulEnv(gym.Env):
    def __init__(self, n_factories=5, n_players=2):
        self.spec = Spec('azul', 10000)
        self.observation_space = spaces.Box(
            low=0,
            # includes first mover tile
            high=len(COLORS),
            shape=(
                # 1 value per tile
                N_TILES + 1,
                # locations of tiles: factory, bag, center, box (n_factories + 3);
                # floor + 25 wall spots + 5 staging rows per player = 31
                n_factories + 3 + (n_players * 31),
            ),
            dtype=np.uint8)

        # location of selection: factories or center
        self.selection_options = n_factories + 1
        # 5 staging rows + floor
        self.placement_options = 6
        self.action_space = spaces.Discrete(
            self.selection_options * self.placement_options * len(COLORS))
        self.azs = AZS(n_players, N_TILES)
        self.active_player = 0
        self.num_players = n_players

    def step(self, a):
        reward = 0.0
        # 0, 0, 0 -> 0
        # 0, 0, 1 -> 1
        # 0, 0, 2 -> 2
        # ...
        # 0, 0, 4 -> 4
        # 0, 1, 0 -> 5
        # 0, 1, 1 -> 6
        # ...
        # 0, 1, 4 -> 9
        print("Action: {}".format(a))
        color = a % len(COLORS)
        color_placements = ((a - color) / len(COLORS))
        placement = color_placements % self.placement_options
        selection = ((color_placements) /
                     self.placement_options) % self.selection_options
        reward = self.azs.act(int(selection), int(color), int(placement),
                              self.active_player)
        # update whose turn it is
        self.active_player = (self.active_player + 1) % self.num_players
        return self.azs.get_obs(), reward, self.azs.game_over()

    def get_obs(self):
        return self.azs.get_obs()

    def reset(self):
        self.azs.reset_game()
        return self.azs.get_obs()
