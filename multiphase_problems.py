import torch

from pina.problem import TimeDependentProblem, SpatialProblem
from pina.operators import grad, laplacian, div
from pina import Condition, LabelTensor
from pina.geometry import CartesianDomain
from pina.equation import Equation,SystemEquation
from pina.equation.equation_factory import FixedValue
from pina.equation import Equation

############ 1-dimensional problems ###########

class AdvectionInterface(TimeDependentProblem, SpatialProblem):

    # time
    T = 0.5
    scale = 0.00
    eps = 0.005

    # assign output/ spatial and temporal variables
    output_variables = ['phi']
    spatial_domain = CartesianDomain({'x': [0, 1]})
    temporal_domain = CartesianDomain({'t': [0, T]})

    @staticmethod
    def phi_initial(input_):
        eps = AdvectionInterface.eps
        x = input_.extract(['x']) - 0.25
        f = lambda x : 0.5 * (1 + torch.tanh(x/(2.0*eps)))
        return f(x)

        # define initial condition
    def initial_condition(input_, output_):
        phi_expected = AdvectionInterface.phi_initial(input_)
        return output_.extract(['phi']) - phi_expected

        # define true solution
    def truth_solution(self, pts):
        eps = AdvectionInterface.eps
        x = pts.extract(['x']) - 0.25
        t = pts.extract(['t'])
        f = lambda x : 0.5 * (1 + torch.tanh(x/(2.0*eps)))
        return f(x-t)

    def advection(input_, output_):
        gradient  = grad(output_, input_)
        dphi_t     = gradient.extract(['dphidt'])
        dphi_x = gradient.extract(['dphidx'])
        eps = AdvectionInterface.eps
        # compute residuals
        normals = dphi_x/(torch.abs(dphi_x)+1e-10)
        normals = normals.detach()
        grad2 = -output_*(1.0-output_)*normals + eps*dphi_x
        grad2 = LabelTensor(grad2, labels=dphi_x.labels)
        sharp = grad(grad2, input_, d=['x'])
        allen_cahn = LabelTensor(sharp, labels=dphi_t.labels) * AdvectionInterface.scale
        return (dphi_t + dphi_x - allen_cahn)


    # problem condition statement
    conditions = {
        'gamma': Condition(
            location=CartesianDomain({'x': 1, 't' : [0, T]}),
            equation=FixedValue(1.0)),
        't0': Condition(location=CartesianDomain({'x': [0, 1], 't': 0}), equation=Equation(initial_condition)),
        'D': Condition(location=CartesianDomain({'x': [0, 1], 't': [0., T]}), equation= Equation(advection))
    }

class AdvectionInterfaceMass(AdvectionInterface):

    def mass_conservation(input_, output_):
            mask = input_['t']==0
            phi_expected = AdvectionInterface.phi_initial(input_).tensor[mask]
            tot_mass = phi_expected.sum()
            out_mass = [output_[(input_['t'] == t).flatten()].sum() for t in torch.unique(input_['t'])]
            out_mass = torch.stack(out_mass)
            return tot_mass - out_mass

    # problem condition statement
    conditions = {}
    for key, condition in AdvectionInterface.conditions.items():
         conditions[key] = condition

    conditions['mass'] = Condition(location=CartesianDomain({'x': [0, 1], 't': [0., AdvectionInterface.T]}), equation= Equation(mass_conservation))