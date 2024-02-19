import torch

from pina import Trainer
from pina.solvers import PINN
from pina.plotter import Plotter
from pina.model import FeedForward
from multiphase_problems import RotatingBubble, RotatingBubbleMass
from utils import create_parser
from pytorch_lightning.callbacks import Callback

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
    
# Hard Network definition
class HardNet(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = FeedForward(*args, **kwargs)

    def forward(self, x):
        force = (x['x']+0.5) * (x['y']+0.5) * (0.5-x['x']) * (0.5-x['y'])
        return self.model(x)*force
    
# Allen Cahn coefficient update
class UpdateCoeff_AllenCahn(Callback):
    def on_train_epoch_end(self, trainer, __):
        print(trainer.solver.problem.__class__.scale)
        trainer.solver.problem.__class__.scale = min(1.0, trainer.current_epoch*1e-4)
        return

if __name__ == "__main__":

    # get parser
    args = create_parser()
    
    # define problem + discretize
    problem = RotatingBubbleMass() if args.mass else RotatingBubble()
    nx = 20
    nt = 100
    problem.discretise_domain(nx, 'grid', locations=['t0'])
    problem.discretise_domain(nx, 'grid', locations=['D'], variables=['x','y'])
    problem.discretise_domain(nt, 'grid', locations=['D'], variables=['t'])
    if args.mass:
        problem.discretise_domain(nx, 'grid'  , locations=['mass'], variables=['x','y'])
        problem.discretise_domain(nt, 'grid'  , locations=['mass'], variables=['t'])

    # extra features
    callback = [UpdateCoeff_AllenCahn()] if args.allen_cahn else []
    
    # make model
    model = HardNet(input_dimensions=3, output_dimensions=1, func=torch.nn.Tanh, inner_size=20, n_layers=4)
    if args.load_model: # loading if not training
        model.load_state_dict(torch.load(args.load_model+'.pth'))

    # make solver
    solver = PINN(
                problem=problem,
                model=model,
                optimizer=torch.optim.AdamW,
                optimizer_kwargs={'lr' : args.lr})

    # make the trainer
    trainer = Trainer(solver=solver, callbacks=callback, max_epochs=args.epochs)

    # train or load
    if args.save_model:
        # train model and save
        trainer.train()
        torch.save(model.state_dict(), args.save_model+'.pth')

    # plot if needes
    if args.plot:
        pl = Plotter()
        pl.plot(solver=solver)

#        T = 2.0*torch.pi
#        times = torch.linspace(0, T, 100)
#        for i, t in enumerate(times):
#            print(t)
#            filename = "phi_" + str(i) + ".pdf"
#            pl.plot(solver=solver, components=['phi'], fixed_variables={'t':t}, filename=filename)
