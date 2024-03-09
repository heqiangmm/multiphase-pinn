import torch
import numpy as np

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
        't0': Condition(location=CartesianDomain({'x': [0, 1], 't': 0}), 
                        equation=Equation(initial_condition)),
        'D': Condition(location=CartesianDomain({'x': [0, 1], 't': [0., T]}), 
                       equation= Equation(advection))
    }

class RotatingBubble(TimeDependentProblem, SpatialProblem):

    # time
    T = 2.0*torch.pi
    scale = 0.00
    eps = 0.005

    # assign output/ spatial and temporal variables
    output_variables = ['phi']
    spatial_domain = CartesianDomain({'x': [-0.5, 0.5], 'y': [-0.5, 0.5]})
    temporal_domain = CartesianDomain({'t': [0, T]})
    center = LabelTensor(torch.tensor([[0.0, 0.0]]), labels=['x', 'y'])
    
    def phi_initial(input_):
        eps = RotatingBubble.eps
        x = input_.extract(['x'])
        y = input_.extract(['y']) - 0.25
        norm_ = torch.sqrt(x**2 + y**2)
        phi_expected = 0.5*(1. + torch.tanh((norm_ - 0.15) / (2.*eps)))
        return -phi_expected + 1.

    # define initial condition
    def initial_condition(input_, output_):
        phi_expected = RotatingBubble.phi_initial(input_)
        return output_.extract(['phi']) - phi_expected

    def advection(input_, output_):
        gradient  = grad(output_, input_)
        dphi_t    = gradient.extract(['dphidt'])
        dphi_x    = gradient.extract(['dphidx'])
        dphi_y    = gradient.extract(['dphidy'])
        x = input_.extract(['x'])
        y = input_.extract(['y'])
        t = input_.extract(['t'])
        eps = RotatingBubble.eps
        # compute residuals
        u1    = - y
        u2    =   x
        u_max = np.sqrt(2.0*0.5**2)
        normals1 = dphi_x/(torch.sqrt(dphi_x.pow(2)+dphi_y.pow(2)+1e-16)+1e-10)
        normals2 = dphi_y/(torch.sqrt(dphi_x.pow(2)+dphi_y.pow(2)+1e-16)+1e-10)
        normals1 = normals1.detach()
        normals1 = normals2.detach()
        sharp1 = (-output_*(1.0-output_)*normals1 + eps*dphi_x)*u_max
        sharp2 = (-output_*(1.0-output_)*normals2 + eps*dphi_y)*u_max
        sharp1 = LabelTensor(sharp1, labels=dphi_t.labels)
        sharp2 = LabelTensor(sharp2, labels=dphi_t.labels)
        sharp1 = grad(sharp1, input_, d=['x'])
        sharp2 = grad(sharp2, input_, d=['y'])
        sharp1 = LabelTensor(sharp1, labels=dphi_t.labels)
        sharp2 = LabelTensor(sharp2, labels=dphi_t.labels)
        return dphi_t + u1*dphi_x + u2*dphi_y - RotatingBubble.scale*(sharp1+sharp2)

    # problem condition statement
    conditions = {
        't0': Condition(
            location=CartesianDomain({'x': [-0.5, 0.5], 
                                      'y': [-0.5, 0.5], 
                                      't': 0}), 
            equation=Equation(initial_condition)),
        'D': Condition(
            location=CartesianDomain({'x': [-0.5, 0.5], 
                                      'y': [-0.5, 0.5], 
                                      't': [0, T]}),
            equation=Equation(advection)),
        'gamma1': Condition(
            location=CartesianDomain({'x': -0.5,
                                      'y': [-0.5, 0.5],
                                      't': [0, T]}),
            equation=FixedValue(0.0)),
        'gamma2': Condition(
            location=CartesianDomain({'x': 0.5,
                                      'y': [-0.5, 0.5],
                                      't' : [0, T]}),
            equation=FixedValue(0.0)),
        'gamma3': Condition(
            location=CartesianDomain({'x': [-0.5, 0.5],
                                      'y': -0.5,
                                      't: [0, T]}),
            equation=FixedValue(0.0)),
        'gamma4': Condition(
            location=CartesianDomain({'x': [-0.5, 0.5],
                                      'y': 0.5,
                                      't': [0, T]}),
            equation=FixedValue(0.0)),
    }

class RotatingBubbleMass(RotatingBubble):

    def mass_conservation(input_, output_):
        mask = input_['t']==0
        phi_expected = RotatingBubble.phi_initial(input_).tensor[mask]
        tot_mass = phi_expected.mean()
        out_mass = [output_[(input_['t'] == t).flatten()].mean() 
                    for t in torch.unique(input_['t'])]
        out_mass = torch.stack(out_mass)
        return tot_mass - out_mass

    # problem condition statement
    conditions = {}
    for key, condition in RotatingBubble.conditions.items():
         conditions[key] = condition

    conditions['mass'] = Condition(
        location=CartesianDomain({'x': [-0.5, 0.5], 
                                  'y': [-0.5, 0.5], 
                                  't': [0, RotatingBubble.T]}), 
        equation= Equation(mass_conservation))
        


class RiderKothe(TimeDependentProblem, SpatialProblem):

    # time
    T = 1.0
    scale = 0.00
    eps = 0.005

    # assign output/ spatial and temporal variables
    output_variables = ['phi']
    spatial_domain = CartesianDomain({'x': [-0.5, 0.5], 'y': [-0.5, 0.5]})
    temporal_domain = CartesianDomain({'t': [0, T]})
    center = LabelTensor(torch.tensor([[0.0, 0.0]]), labels=['x', 'y'])
    
    def phi_initial(input_):
        eps = RotatingBubble.eps
        x = input_.extract(['x'])
        y = input_.extract(['y']) - 0.25
        norm_ = torch.sqrt(x**2 + y**2)
        phi_expected = 0.5*(1. + torch.tanh((norm_ - 0.15) / (2.*eps)))
        return -phi_expected + 1.

    # define initial condition
    def initial_condition(input_, output_):
        phi_expected = RiderKothe.phi_initial(input_)
        return output_.extract(['phi']) - phi_expected

    def advection(input_, output_):
        gradient  = grad(output_, input_)
        dphi_t    = gradient.extract(['dphidt'])
        dphi_x    = gradient.extract(['dphidx'])
        dphi_y    = gradient.extract(['dphidy'])
        x = input_.extract(['x'])
        y = input_.extract(['y'])
        t = input_.extract(['t'])
        eps = RiderKothe.eps
        # compute residuals
        u1    = -torch.sin(torch.pi*(x+0.5)).pow(2)*torch.sin(2.*torch.pi*(y+0.5))*torch.cos(torch.pi*t/1.0)
        u2    =  torch.sin(torch.pi*(y+0.5)).pow(2)*torch.sin(2.*torch.pi*(x+0.5))*torch.cos(torch.pi*t/1.0)
        u_max = torch.abs(torch.cos(torch.pi*t/1.0))
        normals1 = dphi_x/(torch.sqrt(dphi_x.pow(2)+dphi_y.pow(2)+1e-16)+1e-10)
        normals2 = dphi_y/(torch.sqrt(dphi_x.pow(2)+dphi_y.pow(2)+1e-16)+1e-10)
        normals1 = normals1.detach()
        normals1 = normals2.detach()
        sharp1 = (-output_*(1.0-output_)*normals1 + eps*dphi_x)*u_max
        sharp2 = (-output_*(1.0-output_)*normals2 + eps*dphi_y)*u_max
        sharp1 = LabelTensor(sharp1, labels=dphi_t.labels)
        sharp2 = LabelTensor(sharp2, labels=dphi_t.labels)
        sharp1 = grad(sharp1, input_, d=['x'])
        sharp2 = grad(sharp2, input_, d=['y'])
        sharp1 = LabelTensor(sharp1, labels=dphi_t.labels)
        sharp2 = LabelTensor(sharp2, labels=dphi_t.labels)
        return dphi_t + u1*dphi_x + u2*dphi_y - RiderKothe.scale*(sharp1+sharp2)

    # problem condition statement
    conditions = {
        't0': Condition(
            location=CartesianDomain({'x': [-0.5, 0.5], 
                                      'y': [-0.5, 0.5], 
                                      't': 0}), 
            equation=Equation(initial_condition)),
        'D': Condition(
            location=CartesianDomain({'x': [-0.5, 0.5], 
                                      'y': [-0.5, 0.5], 
                                      't': [0, T]}),
            equation=Equation(advection)),
        'gamma1': Condition(
            location=CartesianDomain({'x': -0.5,
                                      'y': [-0.5, 0.5],
                                      't': [0, T]}),
            equation=FixedValue(0.0)),
        'gamma2': Condition(
            location=CartesianDomain({'x': 0.5,
                                      'y': [-0.5, 0.5],
                                      't' : [0, T]}),
            equation=FixedValue(0.0)),
        'gamma3': Condition(
            location=CartesianDomain({'x': [-0.5, 0.5],
                                      'y': -0.5,
                                      't: [0, T]}),
            equation=FixedValue(0.0)),
        'gamma4': Condition(
            location=CartesianDomain({'x': [-0.5, 0.5],
                                      'y': 0.5,
                                      't': [0, T]}),
            equation=FixedValue(0.0)),
    }

class RiderKotheMass(RiderKothe):

    def mass_conservation(input_, output_):
        mask = input_['t']==0
        phi_expected = RiderKothe.phi_initial(input_).tensor[mask]
        tot_mass = phi_expected.mean()
        out_mass = [output_[(input_['t'] == t).flatten()].mean() 
                    for t in torch.unique(input_['t'])]
        out_mass = torch.stack(out_mass)
        return tot_mass - out_mass

    # problem condition statement
    conditions = {}
    for key, condition in RiderKothe.conditions.items():
         conditions[key] = condition

    conditions['mass'] = Condition(
        location=CartesianDomain({'x': [-0.5, 0.5], 
                                  'y': [-0.5, 0.5], 
                                  't': [0, RiderKothe.T]}), 
        equation= Equation(mass_conservation))
