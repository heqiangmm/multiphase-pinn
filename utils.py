import argparse
import torch.nn as nn
import dill                             # for saving py objects
import glob                             # i/o operations
import os                               # i/o operations
import imageio                          # for gif
import matplotlib.pyplot as plt         # for plotting
import seaborn as sns
import torch

from pina import LabelTensor


plt.rcParams['text.latex.preamble']=r"\usepackage{lmodern}"
plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 14
plt.rcParams['font.family'] = 'lmodern'

class PostProcessing(object):
    def __init__(self, trainer, evaluation_pts=None):
        self.trainer = trainer
        self.solver = trainer.solver
        self.problem = self.solver.problem
        if evaluation_pts is not None:
            # some checks
            self._check_pts(evaluation_pts)
            self.pts = evaluation_pts
            self.sol = self.solver(self.pts)

    def plot_thickness(self, pts=None, type = 'grad', filename=None, **kwargs):
        from pina.operators import grad
        assert type in ['grad', 'direct']
        pts  = self._check_pts_to_use(pts)
        pts = pts.requires_grad_()
        x, times =  self._divide_pts_time_space(pts)
        surface = []
        for t in times:
            t = t.reshape(-1, 1)
            t.labels = times.labels
            pts = x.append(t, mode='cross')
            sol = self.solver(pts)
            grad_sol = grad(sol, pts, d=['x', 'y'])
            surface.append(grad_sol.pow(2).sum().sqrt().detach())
        # plot
        plt.plot(times.detach(), surface, color='orange', **kwargs)
        plt.xlabel('Time')
        plt.ylabel('Surface')
        # Adjust layout
        plt.tight_layout()

        if filename is not None:
            plt.savefig(filename+'.pdf')
        else:
            plt.show()

    def plot_mass(self, pts=None, filename=None, **kwargs):
        pts = self._check_pts_to_use(pts)
        x, times =  self._divide_pts_time_space(pts)
        mass = []
        for t in times:
            t = t.reshape(-1, 1)
            t.labels = times.labels
            pts = x.append(t, mode='cross')
            sol = self.solver(pts).detach()
            mass.append(sol.mean())
        # plot
        mass = torch.stack(mass)
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        # Plot histogram with KDE on the left
        sns.histplot(mass, kde=True,color='orange', stat='density', element="step", ax=axs[0], **kwargs)
        axs[0].set_xlabel('Mass')
        # Plot sorted samples on the right
        axs[1].plot(times, mass, color='orange')
        axs[1].set_xlabel('Time')
        axs[1].set_ylabel('Mass')
        # Adjust layout
        plt.tight_layout()

        if filename is not None:
            plt.savefig(filename+'.pdf')
        else:
            plt.show()

    def plot_solution(self, pts=None, filename=None, format='pdf', **kwargs):
        pts = self._check_pts_to_use(pts)
        x, times =  self._divide_pts_time_space(pts)
        for idx, t in enumerate(times):
            t = t.reshape(-1, 1)
            t.labels = times.labels
            pts = x.append(t, mode='cross')
            sol = self.solver(pts).detach()
            plt.tricontourf(pts.extract(pts.labels[0]).flatten(),
                            pts.extract(pts.labels[1]).flatten(),
                            sol.flatten(),
                            **kwargs)
            plt.xlabel(pts.labels[0])
            plt.ylabel(pts.labels[1])
            if filename is not None:
                plt.savefig(filename+f'_{idx}.{format}')
            else:
                plt.show()

    # def plot_thickness(self, pts=None, filename=None, type='direct', **kwargs):
    #     assert type in ['grad', 'direct']
    #     pts, _ = self._check_pts_to_use(pts)
    #     x, times =  self._divide_pts_time_space()
    #     thickness = []
    #     for idx, t in enumerate(times):
    #         pts = x.append(t, mode='cross')
    #         sol = self.solver(sol)
                
    def create_gif(self, filename, pts=None, **kwargs):
        # compute plots
        self.plot_solution(pts, 'Aplot', format='png', **kwargs)
        # append images
        images = []
        for file in glob.glob("*.png"):
            images.append(file)
        # sort
        images = sorted(images, key=lambda x: int(x.split('_')[1].split('.')[0]))
        # Create the GIF from the saved images
        with imageio.get_writer(f'{filename}.gif', mode='I') as writer:
            for file in images:
                image = imageio.v2.imread(file)
                writer.append_data(image)
        # rm files
        png_files = glob.glob('Aplot_*')
        for file in png_files:
            os.remove(file)
            
    def _divide_pts_time_space(self, pts):
        # unique t
        t = pts.extract('t').unique(dim=0)
        t.labels = ['t']
        # unique x
        labels = [l for l in pts.labels if l != 't']
        x = pts.extract(labels).unique(dim=0)
        x.labels = labels
        return x, t

    def _check_pts(self, pts):
        # some checks
        assert isinstance(pts, LabelTensor)
        assert sorted(pts.labels) == sorted(self.problem.input_variables)
        assert any([lab=='t' for lab in pts.labels])

    def _check_pts_to_use(self, pts=None):
        # use the evaluation pts given in the init
        if pts is None:
            return self.pts
        self._check_pts(pts)
        return pts
       
    
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

    parser.add_argument("--n_spatial", help="discretization in space", type=int, default=20)
    parser.add_argument("--n_time", help="discretization in time", type=int, default=100)

    parser.add_argument("--n_layers", help="number of layers",
                        type=int, default=4)
    parser.add_argument("--inner_size", help="inner size dimension",
                        type=int, default=20)
    parser.add_argument("--func", help="activation function",
                        type=nn.Module, default=nn.Tanh)

    args = parser.parse_args()
    return args


def save_state(object_, file_name):
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
    return dill.load(open(file_name, "rb"))