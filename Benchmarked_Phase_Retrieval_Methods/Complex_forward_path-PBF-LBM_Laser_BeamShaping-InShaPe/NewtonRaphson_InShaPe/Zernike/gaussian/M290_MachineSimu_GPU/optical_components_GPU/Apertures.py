"""
Shengyuan Yan, TU/e, Eindhoven, Netherlands, 17:00pm, 4/02/2025
"""

from M290_MachineSimu_GPU.complex_field_tools_GPU.Simulation_grid import MeshGrid

import numpy as np
from scipy.special import comb

from LightPipes import *
import numpy as _np
import torch
from torch import nn

def SmoothStep(x, x_min=0, x_max=1, N=0):

    x = torch.clamp((x - x_min) / (x_max - x_min), 0, 1)

    result = 0
    for n in range(0, N + 1):
        comb_Nn_n = comb(N + n, n) # one real number
        comb_2N_Nn = comb(2 * N + 1, N - n) # one real number
        
        result += comb_Nn_n * comb_2N_Nn * (-x) ** n

    result *= x ** (N + 1)

    return result

def SmoothCircAperture(field, R, s, gridsize, x_shift = 0.0, y_shift = 0.0, n = 2, T = 1.0):
    
    device = field.device

    Y, X = MeshGrid(field, gridsize)
    Y = Y - y_shift
    X = X - x_shift

    SqrtT=T**0.5
    
    # Use SmootStep function here
    smooth_edges = SmoothStep(R-torch.sqrt(X*X+Y*Y),-s/2,s/2,n)
    smooth_edges = smooth_edges.to(device)
    
    smooth_edges_amp = torch.sqrt(smooth_edges)
    field *= SqrtT * smooth_edges_amp

    return field

class SmoothEdgeAperture(nn.Module):
    def __init__(self, R, s):
        super(SmoothEdgeAperture, self).__init__()
        self.R = R
        self.s = s

    def forward(self, Fin, gridsize):
        
        Fout = SmoothCircAperture(Fin, self.R, self.s, gridsize)

        return Fout


def CircularAperture(field, R, gridsize, x_shift = 0.0, y_shift = 0.0):
    
    Y, X = MeshGrid(field, gridsize)
    Y = Y - y_shift
    X = X - x_shift
    
    dist_sq = X**2 + Y**2
    
    if field.dim() == 3:
        field[:, dist_sq > R**2] = 0.0
    elif field.dim() == 2:
        field[dist_sq > R**2] = 0.0
    
    return field