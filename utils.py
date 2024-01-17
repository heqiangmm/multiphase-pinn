import argparse

def create_parser():
    parser = argparse.ArgumentParser(description="Run Multiphase Problems")

    # group = parser.add_mutually_exclusive_group(required=True)
    # group.add_argument("--load_model", type=str, help="Path to the model file to load.")
    # group.add_argument("--save_model", type=str, help="Path to save the model.")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--save_model", help="directory to save model", type=str)
    group.add_argument("--load_model", help="directory to load model", type=str)

    parser.add_argument('--allen_cahn', action='store_true', help="adding allen cahn terms during training (default)")
    parser.add_argument('--no-allen_cahn', dest='allen_cahn', action='store_false' , help="removing allen cahn terms during training")
    parser.set_defaults(allen_cahn=True)

    parser.add_argument('--plot', action='store_true', help="plot the solution (default)")
    parser.add_argument('--no-plot', dest='plot', action='store_false', help="don't plot the solution")
    parser.set_defaults(plot=True)

    parser.add_argument("--epochs", help="number of training epochs (default 5000)", type=int, default=5000)
    parser.add_argument("--lr", help="learning rate (default 0.001)", type=float, default=0.001)
    args = parser.parse_args()
    return args
