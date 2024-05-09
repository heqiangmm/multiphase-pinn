import torch
import numpy as np

from pina.problem import TimeDependentProblem, SpatialProblem
from pina.operators import grad, laplacian, div
from pina import Condition, LabelTensor
from pina.geometry import CartesianDomain
from pina.equation import Equation,SystemEquation
from pina.equation.equation_factory import FixedValue
from pina.equation import Equation


class RotatingBubble(TimeDependentProblem, SpatialProblem):

    # time
    T = 2.0*torch.pi
    scale = 0.00
    eps = 0.0005
    x_domain = [-0.5, 0.5]
    y_domain = [-0.5, 0.5]
    t_domain = [0, T]

    # assign output/ spatial and temporal variables
    output_variables = ['phi']
    spatial_domain = CartesianDomain({'x': x_domain, 'y': y_domain})
    temporal_domain = CartesianDomain({'t': t_domain})
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
        'gamma1': Condition(
            location=CartesianDomain({'x': x_domain[0], 
                                      'y': y_domain, 
                                      't': t_domain}), 
            equation=FixedValue(0)),
        'gamma2': Condition(
            location=CartesianDomain({'x': x_domain[1], 
                                      'y': y_domain, 
                                      't': t_domain}), 
            equation=FixedValue(0)),
        'gamma3': Condition(
            location=CartesianDomain({'x': x_domain, 
                                      'y': y_domain[0], 
                                      't': t_domain}), 
            equation=FixedValue(0)),
        'gamma4': Condition(
            location=CartesianDomain({'x': x_domain, 
                                      'y': y_domain[1], 
                                      't': t_domain}), 
            equation=FixedValue(0)),
        't0': Condition(
            location=CartesianDomain({'x': x_domain, 
                                      'y': y_domain, 
                                      't': t_domain[0]}), 
            equation=Equation(initial_condition)),
        'D': Condition(
            location=CartesianDomain({'x': x_domain, 
                                      'y': y_domain, 
                                      't': t_domain}),
            equation=Equation(advection))
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
        location=CartesianDomain({'x': RotatingBubble.x_domain, 
                                  'y': RotatingBubble.y_domain, 
                                  't': RotatingBubble.t_domain}), 
        equation= Equation(mass_conservation))
        

class RiderKothe(TimeDependentProblem, SpatialProblem):

    # time
    T = 1.0
    scale = 0.00
    eps = 0.005
    x_domain = [-0.5, 0.5]
    y_domain = [-0.5, 0.5]
    t_domain = [0, T]

    # assign output/ spatial and temporal variables
    output_variables = ['phi']
    spatial_domain = CartesianDomain({'x': x_domain, 'y': y_domain})
    temporal_domain = CartesianDomain({'t': t_domain})
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
            location=CartesianDomain({'x': x_domain, 
                                      'y': y_domain, 
                                      't': t_domain[0]}), 
            equation=Equation(initial_condition)),
        'D': Condition(
            location=CartesianDomain({'x': x_domain, 
                                      'y': y_domain, 
                                      't': t_domain}),
            equation=Equation(advection)),
        'gamma1': Condition(
            location=CartesianDomain({'x': x_domain[0],
                                      'y': y_domain,
                                      't': t_domain}),
            equation=FixedValue(0.0)),
        'gamma2': Condition(
            location=CartesianDomain({'x': x_domain[1],
                                      'y': y_domain,
                                      't' : t_domain}),
            equation=FixedValue(0.0)),
        'gamma3': Condition(
            location=CartesianDomain({'x': x_domain,
                                      'y': y_domain[0],
                                      't': t_domain}),
            equation=FixedValue(0.0)),
        'gamma4': Condition(
            location=CartesianDomain({'x': x_domain,
                                      'y': y_domain[1],
                                      't': t_domain}),
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
        location=CartesianDomain({'x': RiderKothe.x_domain, 
                                  'y': RiderKothe.y_domain, 
                                  't': RiderKothe.t_domain}), 
        equation= Equation(mass_conservation))

class RayleighTaylor(TimeDependentProblem, SpatialProblem):

    # params
    d = 1.0
    At = 0.5
    Re = 3000.
    gg = 9.82
    rho1 = 1.5
    rho2 = 0.5
    mu = (rho1*d**1.5*gg**0.5)/Re
    g = torch.tensor([[0, 9.82]])
    tstar = np.sqrt(d/(gg*At))
    T = 1.0*tstar    #0.5                             # already adimensional
    scale = 0.00
    eps = 0.02

    # domains
    x_domain = [-0.5, 0.5]
    y_domain = [0., 4.]
    t_domain = [0, T]

    # assign output/ spatial and temporal variables
    output_variables = ['phi', 'ux', 'uy', 'pr']
    spatial_domain = CartesianDomain({'x': x_domain, 'y': y_domain})
    temporal_domain = CartesianDomain({'t': t_domain})

    # define initial condition
    def initial_phi(input_, output_):
        d = RayleighTaylor.d
        eps = RayleighTaylor.eps
        x = input_.extract(['x'])+0.5
        y = input_.extract(['y'])
        f = 2.*d + 0.1*d*torch.cos(torch.pi*2.*x / d)
        phi_expected = 0.5*(1. + torch.tanh( (y-f) / (2.*eps)))
        return output_.extract(['phi']) - phi_expected

    # define initial condition
    def initial_ux(input_, output_):
        return output_.extract(['ux'])

    # define initial condition
    def initial_uy(input_, output_):
        return output_.extract(['uy'])

    # define initial condition
    def initial_p(input_, output_):
        d = RayleighTaylor.d
        eps = RayleighTaylor.eps
        x = input_.extract(['x'])+0.5
        y = input_.extract(['y'])
        f = 2.*d + 0.1*d*torch.cos(torch.pi*2.*x / d)
        phi = 0.5*(1. + torch.tanh( (y-f) / (2.*eps)))
        rho = RayleighTaylor.rho1 * phi + RayleighTaylor.rho2 * (1.-phi)
        grad_p = grad(output_, input_, components=['pr'], d=['x', 'y'])
        g = RayleighTaylor.g
        g = g.to(input_.device)
        return grad_p + rho*g

    def zerodivegence(input_, output_):
        return div(output_, input_, d=['x', 'y'], components=['ux', 'uy'])

    def advection(input_, output_):
        gradient  = grad(output_, input_)
        dphi_t    = gradient.extract(['dphidt'])
        dphi_x    = gradient.extract(['dphidx'])
        dphi_y    = gradient.extract(['dphidy'])
        ux = output_.extract('ux')
        uy = output_.extract('uy')
        # compute residuals
        return dphi_t + ux * dphi_x + uy * dphi_y
    
    def ns(input_, output_):
        # RHS of NS equation
        phi = output_.extract('phi')
        rho = RayleighTaylor.rho1 * phi + RayleighTaylor.rho2 * (1.-phi)
        u_t = grad(output_, input_, components=['ux', 'uy'], d=['t'])
        grad_uxx = grad(output_, input_, components=['ux'], d=['x'])
        grad_uxy = grad(output_, input_, components=['ux'], d=['y'])
        grad_uyx = grad(output_, input_, components=['uy'], d=['x'])
        grad_uyy = grad(output_, input_, components=['uy'], d=['y'])
        ux = output_.extract('ux')
        uy = output_.extract('uy')
        conv_x = ux*grad_uxx + uy*grad_uxy
        conv_y = ux*grad_uyx + uy*grad_uyy
        conv = torch.cat((conv_x, conv_y), dim=1)
        RHS = rho*(u_t + conv)
        # LHS of NS equation
        laplacian_u = laplacian(output_, input_, components=['ux', 'uy'], d=['x', 'y'])
        grad_p = grad(output_, input_, components=['pr'], d=['x', 'y'])
        g = RayleighTaylor.g
        g = g.to(input_.device)
        LHS = -grad_p + RayleighTaylor.mu*laplacian_u - rho*g
        return RHS - LHS
    
    def no_slip_u(input_, output_):
        return output_.extract(['ux', 'uy'])
    
    def no_slip_p(input_, output_):
        return output_.extract('pr')
    
    def slip_u(input_, output_):
        return output_.extract('ux')

    def phi_up(input_, output_):
        return 1.0-output_.extract('phi')

    # problem condition statement
    conditions = {
        't0_phi': Condition(
            location=CartesianDomain({'x': x_domain, 
                                      'y': y_domain, 
                                      't': t_domain[0]}), 
            equation=Equation(initial_phi)),
        't0_ux': Condition(
            location=CartesianDomain({'x': x_domain,
                                      'y': y_domain,
                                      't': t_domain[0]}),
            equation=Equation(initial_ux)),
        't0_uy': Condition(
            location=CartesianDomain({'x': x_domain,
                                      'y': y_domain,
                                      't': t_domain[0]}),
            equation=Equation(initial_uy)),
        'D1': Condition(
            location=CartesianDomain({'x': x_domain, 
                                      'y': y_domain, 
                                      't': t_domain}), 
            equation=Equation(zerodivegence)
        ),
        'D2': Condition(
            location=CartesianDomain({'x': x_domain,
                                      'y': y_domain,
                                      't': t_domain}),
            equation=Equation(advection)
        ),
        'D3': Condition(
            location=CartesianDomain({'x': x_domain,
                                      'y': y_domain,
                                      't': t_domain}),
            equation=Equation(ns)
        ),
        'gammatop_u': Condition(
            location=CartesianDomain({'x': x_domain, 
                                      'y': y_domain[1], 
                                      't': t_domain}), 
            equation=Equation(no_slip_u)
        ),
        'gammatop_p': Condition(
            location=CartesianDomain({'x': x_domain,
                                      'y': y_domain[1],
                                      't': t_domain}),
            equation=Equation(no_slip_p)
        ),
        'gammatop_phi': Condition(
            location=CartesianDomain({'x': x_domain,
                                      'y': y_domain[1],
                                      't': t_domain}),
            equation=Equation(phi_up)
        ),
        'gammabottom': Condition(
            location=CartesianDomain({'x': x_domain, 
                                      'y': y_domain[0], 
                                      't': t_domain}), 
            equation=Equation(no_slip_u)
        ),
        'gammaleft': Condition(
            location=CartesianDomain({'x': x_domain[0], 
                                      'y': y_domain, 
                                      't': t_domain}), 
            equation=Equation(slip_u)
        ),
        'gammaright': Condition(
            location=CartesianDomain({'x': x_domain[1], 
                                      'y': y_domain, 
                                      't': t_domain}), 
            equation=Equation(slip_u)
        )
    }
