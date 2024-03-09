import torch
import numpy as np

from pina import Trainer, LabelTensor
from pina.solvers import PINN
from pina.plotter import Plotter
from pina.model import FeedForward
from multiphase_problems import RotatingBubble, RotatingBubbleMass
from utils import create_parser
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
    
# Hard Network definition
class HardNet(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = FeedForward(*args, **kwargs)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.model(x))
    
# Allen Cahn coefficient update
class UpdateCoeff_AllenCahn(Callback):
    def on_train_epoch_end(self, trainer, __):
        print(trainer.solver.problem.__class__.scale)
        trainer.solver.problem.__class__.scale = 1.0
        return

if __name__ == "__main__":

    # get parser
    args = create_parser()
    
    # define problem + discretize
    problem = RotatingBubbleMass() if args.mass else RotatingBubble()
    nx = 30
    nt = 400
    problem.discretise_domain(nx, 'grid', locations=['t0'])
    problem.discretise_domain(nx, 'grid', locations=['D'], variables=['x','y'])
    problem.discretise_domain(nt, 'grid', locations=['D'], variables=['t'])
    problem.discretise_domain(nx, 'grid', locations=['gamma1'])
    problem.discretise_domain(nx, 'grid', locations=['gamma2'])
    problem.discretise_domain(nx, 'grid', locations=['gamma3'])
    problem.discretise_domain(nx, 'grid', locations=['gamma4'])
    if args.mass:
        problem.discretise_domain(nx, 'grid'  , locations=['mass'], variables=['x','y'])
        problem.discretise_domain(nt, 'grid'  , locations=['mass'], variables=['t'])

    # extra features
    callback = [UpdateCoeff_AllenCahn()] if args.allen_cahn else []
    
    # make model
    model = HardNet(input_dimensions=3, output_dimensions=1, func=torch.nn.Tanh, inner_size=80, n_layers=4)
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

    # train model and save
    trainer.train()
    torch.save(model.state_dict(), args.save_model+'.pth')

    # plot if needes
    if args.plot:
        pl = Plotter()
        pl.plot(solver=solver)

        T = 1.0
        times = torch.linspace(0, T, 100)
        mass = torch.zeros(100)
        lphi = torch.zeros(100)
        bounds = np.zeros(100)
        for i, t in enumerate(times):
            print(t)
            filename = "phi_" + str(i) + ".pdf"
            pl.plot(solver=solver, components=['phi'], fixed_variables={'t':t}, filename=filename)
        
            # mass errors
            fixed_variables={'t':t}
            v = [
            var for var in solver.problem.input_variables
            if var not in fixed_variables.keys()
             ]

            pts = solver.problem.domain.sample(256, 'grid', variables=v)
            fixed_pts = torch.ones(pts.shape[0], len(fixed_variables))
            fixed_pts *= torch.tensor(list(fixed_variables.values()))
            fixed_pts = fixed_pts.as_subclass(LabelTensor)
            fixed_pts.labels = list(fixed_variables.keys())

            pts = pts.append(fixed_pts)
            pts = pts.to(device=solver.device)

            solution  =  solver.forward(pts).extract(['phi']).as_subclass(torch.Tensor)
            dphi_x    = solver.forward(pts).extract(['dphidx']).as_subclass(torch.Tensor)
            dphi_y    = solver.forward(pts).extract(['dphidy']).as_subclass(torch.Tensor)
            
            l_interface = 1.0/(solution*(1.0-solution))
            lphi[i] = l_interface.mean()
            #l_interface = 1.0/torch.sqrt(dphi_x.pow(2)+dphi_y.pow(2))

            mass[i] = solution.mean()
    
            sol_np = solution.detach().numpy()
    
            temp = sol_np
            for j in range(len(sol_np)):
                if ((sol_np[j]<1.0) and (sol_np[j]>0.0)):
                     temp[j] = 0.0
                elif sol_np[j]<0.0:
                    temp[j] = np.abs(temp[j])
                else:
                 temp[j] = np.abs(1.0-temp[j])

            bounds[i] = np.mean(temp)

        # Plotting
        mass = mass.detach().numpy()
        times = times.detach().numpy()
        plt.plot(times, (mass-mass[0])/mass[0])

        # Adding labels and title
        plt.xlabel('time')
        plt.ylabel('mass')

        plt.tight_layout()
        # Save the plot as a PDF file
        plt.savefig('mass.pdf')
        plt.close()

        # Save times and mass to a text file
        data = np.column_stack((times, np.abs((mass-mass[0]))/mass[0]))
        np.savetxt('mass.txt', data, header='time mass', comments='')

        # Plotting
        lphi = lphi.detach().numpy()
        plt.plot(times,lphi/lphi[0])

        # Adding labels and title
        plt.xlabel('time')
        plt.ylabel('interface thickness')

        plt.tight_layout()
        # Save the plot as a PDF file
        plt.savefig('lphi.pdf')
        plt.close()

        # Save times and mass to a text file
        data = np.column_stack((times, lphi/lphi[0]))
        np.savetxt('lphi.txt', data, header='time thickness', comments='')
        
        # Plotting
        plt.plot(times, bounds)

        # Adding labels and title
        plt.xlabel('time')
        plt.ylabel('out of bounds')

        plt.tight_layout()
        # Save the plot as a PDF file
        plt.savefig('bounds.pdf')
        plt.close()

        # Save times and mass to a text file
        data = np.column_stack((times, bounds))
        np.savetxt('bounds.txt', data, header='time bounds', comments='') 
