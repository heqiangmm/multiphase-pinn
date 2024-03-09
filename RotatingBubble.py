import torch
import numpy as np

from pina import Trainer, LabelTensor
from pina.solvers import PINN
from pina.callbacks import MetricTracker

from multiphase_problems import RotatingBubble, RotatingBubbleMass
from utils import create_parser, load_state, save_state
from nn import SigmoidNet

from pytorch_lightning.callbacks import Callback
import matplotlib.pyplot as plt

# ===================================================== #
#                                                       #
#  This script implements the two dimensional linear    #
#  advection droplet problem. The Multiphase class is   #
#  from TimeDependentProblem, SpatialProblem and we     #
#  denote:                                              #
#           u --> field variable                        #
#           x --> spatial variable                      #
#           y --> spatial variable                      #
#           t --> temporal variable                     #
#                                                       #
# ===================================================== #
    
    
# Allen Cahn coefficient update
class UpdateCoeff_AllenCahn(Callback):
    def __init__(self, switch_epoch):
        super().__init__()
        self._switch_epoch = switch_epoch - 1
    def on_train_epoch_end(self, trainer, __):
        if trainer.current_epoch == self._switch_epoch:
            trainer.solver.problem.__class__.scale = 1

if __name__ == "__main__":

    # get parser + variables
    args = create_parser()
    nx = args.n_spatial
    nt = args.n_time
    func = args.func
    inner_size = args.inner_size
    n_layers = args.n_layers
    
    # define problem + discretize
    problem = RotatingBubbleMass() if args.mass else RotatingBubble()
    problem.discretise_domain(nx, 'grid', locations=['t0'])
    problem.discretise_domain(nx, 'grid',
                              locations=['gamma1', 'gamma2', 'gamma3', 'gamma4'],
                              variables=['x', 'y'])
    problem.discretise_domain(nt, 'grid',
                              locations=['gamma1', 'gamma2', 'gamma3', 'gamma4'],
                              variables=['t'])
    problem.discretise_domain(nx, 'grid', locations=['D'], variables=['x','y'])
    problem.discretise_domain(nt, 'grid', locations=['D'], variables=['t'])
    problem.discretise_domain(nx, 'grid', locations=['gamma1'])
    problem.discretise_domain(nx, 'grid', locations=['gamma2'])
    problem.discretise_domain(nx, 'grid', locations=['gamma3'])
    problem.discretise_domain(nx, 'grid', locations=['gamma4'])
    if args.mass:
        problem.discretise_domain(nx,'grid', 
                                  locations=['mass'], variables=['x','y'])
        problem.discretise_domain(nt,'grid',
                                  locations=['mass'], variables=['t'])

    # extra features
    callback = [UpdateCoeff_AllenCahn(args.epochs//2)] if args.allen_cahn else[]
    
    # make model
    model = SigmoidNet(input_dimensions=3, output_dimensions=1,
                       func=func, inner_size=inner_size, n_layers=n_layers)
    
    # make solver
    solver = PINN(
                problem=problem,
                model=model,
                optimizer=torch.optim.Adam,
                optimizer_kwargs={'lr' : args.lr})

    # make the trainer
    if args.save_model:
        trainer = Trainer(solver=solver, callbacks=callback + [MetricTracker()],
                        max_epochs=args.epochs,
                        deterministic=True)
        trainer.train()
        save_state(trainer, args.save_model)
    elif args.load_model:
        trainer = load_state(args.load_model)
