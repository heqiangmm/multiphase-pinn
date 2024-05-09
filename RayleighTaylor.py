import torch
import numpy as np

from pina import Trainer, LabelTensor
from pina.plotter import Plotter
from pina.solvers import PINN
from pina.callbacks import MetricTracker

from multiphase_problems import RayleighTaylor
from utils import create_parser
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
    def on_train_epoch_end(self, trainer, __):
        #print(trainer.solver.problem.__class__.scale)
        trainer.solver.problem.__class__.scale = 0.0
        return

if __name__ == "__main__":

    # get parser + variables
    args = create_parser()
    # define problem + discretize
    nx = args.n_spatial
    ny = 4*nx
    nt = args.n_time
    func = args.func
    inner_size = args.inner_size
    n_layers = args.n_layers
    problem = RayleighTaylor() if args.mass else RayleighTaylor()
    problem.discretise_domain(nx, 'grid', locations=['t0_phi'], variables=['x'])
    problem.discretise_domain(ny, 'grid', locations=['t0_phi'], variables=['y', 't'])
    problem.discretise_domain(nx, 'grid', locations=['t0_ux'], variables=['x'])
    problem.discretise_domain(ny, 'grid', locations=['t0_ux'], variables=['y', 't'])
    problem.discretise_domain(nx, 'grid', locations=['t0_uy'], variables=['x'])
    problem.discretise_domain(ny, 'grid', locations=['t0_uy'], variables=['y', 't'])
    problem.discretise_domain(nx, 'grid',
                              locations=['gammatop_u','gammatop_p','gammatop_phi', 'gammabottom'],
                              variables=['x','y'])
    problem.discretise_domain(ny, 'grid',
                              locations=['gammaleft', 'gammaright'],
                              variables=['x','y'])
    problem.discretise_domain(nt, 'grid',
                              locations=['gammatop_u','gammatop_p','gammatop_phi', 'gammabottom', 'gammaleft', 'gammaright'],
                              variables=['t'])
    problem.discretise_domain(nx, 'grid', locations=['D1'], variables=['x'])
    problem.discretise_domain(ny, 'grid', locations=['D1'], variables=['y'])
    problem.discretise_domain(nt, 'grid', locations=['D1'], variables=['t'])
    problem.discretise_domain(nx, 'grid', locations=['D2'], variables=['x'])
    problem.discretise_domain(ny, 'grid', locations=['D2'], variables=['y'])
    problem.discretise_domain(nt, 'grid', locations=['D2'], variables=['t'])
    problem.discretise_domain(nx, 'grid', locations=['D3'], variables=['x'])
    problem.discretise_domain(ny, 'grid', locations=['D3'], variables=['y'])
    problem.discretise_domain(nt, 'grid', locations=['D3'], variables=['t'])
    problem.discretise_domain(nx, 'grid', locations=['gammatop_u','gammatop_p','gammatop_phi',])
    problem.discretise_domain(nx, 'grid', locations=['gammabottom'])
    problem.discretise_domain(ny, 'grid', locations=['gammaleft'])
    problem.discretise_domain(ny, 'grid', locations=['gammaright'])
    # define problem + discretize
    if args.mass:
        problem.discretise_domain(nx,'grid', 
                                  locations=['mass'], variables=['x','y'])
        problem.discretise_domain(nt,'grid',
                                  locations=['mass'], variables=['t'])

    # extra features
    callback = [UpdateCoeff_AllenCahn()] if args.allen_cahn else[]
    
    # make model
    model = SigmoidNet(input_dimensions=3, output_dimensions=4,
                       func=func, inner_size=inner_size, n_layers=n_layers)

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
    
    if args.plot:
        d=1.0
        gg=9.82
        At=0.5
        tstar = np.sqrt(d/(gg*At))
        T = 0.5*tstar
        times = torch.linspace(0, T, 100)
        mass = torch.zeros(100)
        yplus= np.zeros(100)
        yminus= np.zeros(100)
        lphi = torch.zeros(100)
        bounds = np.zeros(100)
        var = ['phi','ux','uy','pr']
        for v in range(4):
             pl = Plotter()
             pl.plot(solver=solver, components=[var[v]])
             for i, t in enumerate(times):
                 print(t)
                 filename = var[v] + "_" + str(i) + ".pdf"
                 pl.plot(solver=solver, components=[var[v]], fixed_variables={'t':t}, filename=filename)

                 if v==1:
                    fixed_variables={'t':t}
                    vv = [
                    var for var in solver.problem.input_variables
                    if var not in fixed_variables.keys()
                    ]

                    pts = solver.problem.domain.sample(512, 'grid', variables=vv)
                    fixed_pts = torch.ones(pts.shape[0], len(fixed_variables))
                    fixed_pts *= torch.tensor(list(fixed_variables.values()))
                    fixed_pts = fixed_pts.as_subclass(LabelTensor)
                    fixed_pts.labels = list(fixed_variables.keys())

                    pts = pts.append(fixed_pts)
                    pts = pts.to(device=solver.device)

                    solution  =  solver.forward(pts).extract(['phi']).as_subclass(torch.Tensor)
                    l_interface = (solution)*(1.0-solution)
                    lphi[i] = l_interface.mean()

                    mass[i] = solution.mean()
    
                    sol_np = solution.detach().numpy()
                    temp = sol_np
                
                    yy = pts.extract('y')
                    yy_np = yy.detach().numpy()

                    yplus[i] =-1e10
                    yminus[i]=1e10
                    for j in range(len(sol_np)):
                        if sol_np[j]<0.5:
                            yplus[i]=max(yplus[i],yy_np[j])
                        else:
                            yminus[i]=min(yminus[i],yy_np[j])

             if v==1:
                # Plotting
                mass = mass.detach().numpy()
                times = times.detach().numpy()
                plt.plot(times/tstar, (mass-mass[0])/mass[0])

                # Adding labels and title
                plt.xlabel('time')
                plt.ylabel('mass')

                plt.tight_layout()
                # Save the plot as a PDF file
                plt.savefig('mass.pdf')
                plt.close()

                # Save times and mass to a text file
                data = np.column_stack((times/tstar, np.abs((mass-mass[0]))/mass[0]))
                np.savetxt('mass.txt', data, header='time mass', comments='')

                # Plotting
                lphi = lphi.detach().numpy()
                plt.plot(times/tstar,lphi/lphi[0])

                # Adding labels and title
                plt.xlabel('time')
                plt.ylabel('interface thickness')

                plt.tight_layout()
                # Save the plot as a PDF file
                plt.savefig('lphi.pdf')
                plt.close()

                # Save times and mass to a text file
                data = np.column_stack((times/tstar, lphi/lphi[0]))
                np.savetxt('lphi.txt', data, header='time thickness', comments='')

                # Plotting
                plt.plot(times/tstar, yplus)
                plt.plot(times/tstar, yminus)

                # Adding labels and title
                plt.xlabel('time')

                plt.tight_layout()
                # Save the plot as a PDF file
                plt.savefig('plumes.pdf')
                plt.close()

                # Save times and mass to a text file
                data = np.column_stack((times/tstar, yplus, yminus))
                np.savetxt('plumes.txt', data, header='time yplus yminus', comments='')
