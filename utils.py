import argparse
import torch.nn as nn
import dill


def create_parser():
    parser = argparse.ArgumentParser(description="Run Multiphase Problems")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--save_model", help="directory to save model", type=str)
    group.add_argument("--load_model", help="directory to load model", type=str)

    parser.add_argument('--allen_cahn',
                        action='store_true',
                        help="adding allen cahn terms in training (default)")
    parser.add_argument('--no-allen_cahn', dest='allen_cahn',
                        action='store_false'
                        , help="removing allen cahn terms during training")
    parser.set_defaults(allen_cahn=True)

    parser.add_argument('--plot', action='store_true',
                        help="plot the solution (default)")
    parser.add_argument('--no-plot', dest='plot', action='store_false',
                        help="don't plot the solution")
    parser.set_defaults(plot=True)

    parser.add_argument('--mass', action='store_true',
                        help="don't apply mass conservation")
    parser.add_argument('--no-mass', dest='mass', action='store_false',
                        help="apply mass conservation (default)")
    parser.set_defaults(mass=False)

    parser.add_argument("--epochs",
                        help="number of training epochs (default 5000)",
                        type=int, default=5000)
    parser.add_argument("--lr", help="learning rate (default 0.001)",
                        type=float, default=0.001)

    parser.add_argument("--n_spatial", help="discretization in space", type=float, default=20)
    parser.add_argument("--n_time", help="discretization in time", type=float, default=100)

    parser.add_argument("--n_layers", help="number of layers",
                        type=float, default=4)
    parser.add_argument("--inner_size", help="inner size dimension",
                        type=float, default=20)
    parser.add_argument("--func", help="activation function",
                        type=nn.Module, default=nn.Tanh)

    args = parser.parse_args()
    return args


def save_state(object_, file_name="pso"):
    """
    Saving python object as a pkl file
    """
    with open(file_name + ".pkl", "wb") as file_:
        dill.dump(object_, file_)

def load_state(file_name):
    """
    Load object saved attributes from file_name
    """
    import dill
    return dill.load(open(file_name + ".pkl", "rb"))