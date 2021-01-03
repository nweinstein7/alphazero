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

from scipy.stats import pearsonr
from sklearn.utils import shuffle

PATH = "./model.pt"
""" CNN representing estimated value for each board state.
"""


class Net(TrainableModel):
    def __init__(self):

        super(TrainableModel, self).__init__()
        self.conv = nn.Conv2d(9, 32, kernel_size=(3, 3), padding=(1, 1))
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
        print(f"Target size: {Y_target.size()}")
        Y_target = Y_target.view(-1, 1)
        loss = F.mse_loss(Y_pred, Y_target)
        print(f"LOSS: {loss}")
        return (loss)

    def forward(self, x):
        x = x['input']
        x = x.permute(0, 3, 1, 2)
        print(f"Initial size: {x.size()}")
        x = F.relu(self.conv(x))
        print(f"Size after conv: {x.size()}")
        x = F.relu(self.conv2(x))
        print(f"Size after conv2: {x.size()}")

        x = F.dropout(x, p=0.2, training=self.training)
        print("Done dropout")
        x = F.relu(self.conv3(x))
        print("Done relu")
        x = F.relu(self.conv4(x))
        print("Done relu2")
        x = x.mean(dim=2).mean(dim=2)
        print("Done mean")
        x = self.linear(x)
        print("Done linear")
        return {'target': x}

    def checkpoint(self):
        torch.save(
            {
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, PATH)

    def maybe_load_from_file(self, path=PATH):
        if os.path.exists(path):
            print(f"Found model to load: {path}")
            checkpoint = None
            if torch.cuda.is_available():
                checkpoint = torch.load(PATH)
            else:
                checkpoint = torch.load(PATH, map_location=torch.device('cpu'))
            self.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def score(self, preds, targets):
        print(f"PREDS: {preds['target'][:, 0]}")
        print(f"TARGETS: {targets['target']}")

        #score, _ = pearsonr(preds['target'][:, 0], targets['target'])
        #base_score, _ = pearsonr(preds['target'][:, 0],
        #                         shuffle(targets['target']))
        #print(f"Score: {score}. Base score: {base_score}")
        return f"pred: {preds['target'][:, 0]} actual: {targets['target']}"


if __name__ == "__main__":
    # Pytorch can deadlock without this
    # https://pytorch.org/docs/stable/notes/multiprocessing.html
    multiprocessing.set_start_method('spawn')
    manager = Manager()
    model = Net()
    model.compile(torch.optim.Adadelta, lr=0.3)

    model.maybe_load_from_file()
    controller = AzulController(manager, model)

    with Pool(processes=2) as pool:
        for i in range(0, 1000):
            game = AzulSimulator(2)
            game.load(game)
            while not game.over():
                game.make_move(
                    controller.best_move(game, playouts=10, pool=pool))
                game.print_board()
                print()
