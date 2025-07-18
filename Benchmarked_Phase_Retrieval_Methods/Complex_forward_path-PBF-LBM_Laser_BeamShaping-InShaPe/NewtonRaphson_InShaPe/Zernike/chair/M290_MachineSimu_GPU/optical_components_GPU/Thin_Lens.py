"""
Shengyuan Yan, TU/e, Eindhoven, Netherlands, 17:00pm, 4/02/2025
"""

from M290_MachineSimu_GPU.complex_field_tools_GPU.Simulation_grid import MeshGrid
from M290_MachineSimu_GPU.complex_field_tools_GPU.complex_field_tools import *

import numpy as _np
import torch
from torch import nn

def thin_lens(field, f, gridsize, wavelength, x_shift = 0.0, y_shift = 0.0):

    _2pi = 2*_np.pi
    legacy=True
    if legacy:
        _2pi = 3.1415926*2
    k = _2pi/wavelength
    
    yy, xx = MeshGrid(field, gridsize)
    xx -= x_shift
    yy -= y_shift
    fi = -k*(xx**2+yy**2)/(2*f)
    field *= torch.exp(1j * fi)
    
    return field

class ThinLens(nn.Module):
    def __init__(self, wavelength):
        super(ThinLens, self).__init__()
        self.wavelength = wavelength

    def forward(self, Fin, f, gridsize):
        
        Fout = thin_lens(Fin, f, gridsize, self.wavelength)

        return Fout