import sys, os, random, time

from itertools import product
import numpy as np
import IPython

from modules import TrainableModel
from games.azul.azul_simulator import AzulSimulator

import multiprocessing
from multiprocessing import Pool, Manager

from games.azul.azul_controller import AzulController
from games.azul.azul_simulator import AzulSimulator

import torch
import torch.nn as nn
import torch.nn.functional as F

PATH = "./model.pt"
""" CNN representing estimated value for each board state.
"""


class Net(TrainableModel):
    def __init__(self):

        super(TrainableModel, self).__init__()
        self.conv = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1))
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1))

        self.conv3 = nn.Conv2d(32,
                               64,
                               kernel_size=(3, 3),
                               padding=(1, 1),
                               dilation=2)
        self.conv4 = nn.Conv2d(64,
                               64,
                               kernel_size=(3, 3),
                               padding=(2, 2),
                               dilation=2)
        self.linear = nn.Linear(64, 1)

    def loss(self, data, data_pred):
        Y_pred = data_pred["target"]
        Y_target = data["target"]
        loss = F.mse_loss(Y_pred, Y_target)
        print(f"LOSS: {loss}")
        return (loss)

    def forward(self, x):
        x = x['input']
        print(f"Initial size: {x.size()}")
        x = F.relu(self.conv(x))
        print(f"Size after conv: {x.size()}")
        x = F.relu(self.conv2(x))
        print(f"Size after conv2: {x.size()}")

        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.mean(dim=2).mean(dim=2)
        x = self.linear(x)
        return {'target': x}

    def checkpoint(self):
        torch.save(
            {
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, PATH)

    def maybe_load_from_file(self):
        if os.path.exists(PATH):
            print(f"Found model to load: {PATH}")
            checkpoint = torch.load(PATH)
            self.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


if __name__ == "__main__":

    manager = Manager()
    model = Net()
    model.compile(torch.optim.Adadelta, lr=0.3)

    model.maybe_load_from_file()
    controller = AzulController(model)

    for i in range(0, 1000):
        game = AzulSimulator(2)
        game.load(game)
        game.make_move(random.choice(game.valid_moves()))
        game.print_board()
        print()

        while not game.over():
            game.make_move(controller.best_move(game, playouts=100))
            game.print_board()
            print()