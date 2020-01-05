from games.azul.azul_simulator import AzulSimulator
import pickle
import os

HERE = os.path.dirname(__file__)


def generate_pickle():
    """
    Overwrite the golden azul simulator for testing
    """
    azs = AzulSimulator(2)
    azs.load(azs)
    with open(os.path.join(HERE, 'azs.pkl'), 'wb') as pkl:
        pickle.dump(azs, pkl)


if __name__ == '__main__':
    generate_pickle()