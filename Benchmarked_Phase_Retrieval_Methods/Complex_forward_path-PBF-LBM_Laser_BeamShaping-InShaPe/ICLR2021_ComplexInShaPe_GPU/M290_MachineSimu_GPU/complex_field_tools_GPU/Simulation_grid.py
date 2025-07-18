"""
Shengyuan Yan, TU/e, Eindhoven, Netherlands, 17:00pm, 4/02/2025
"""

import numpy as _np
import torch
from torch import nn

def MeshGrid(field, gridsize):
    
    device = field.device
    
    N = field.shape[-1]
    h, w = N, N
    cy, cx = h // 2, w // 2
    dx, dy = gridsize/N, gridsize/N
    
    Y, X = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    Y = (Y - cy) * dy
    X = (X - cx) * dx
    
    Y = Y.to(device)
    X = X.to(device)
    
    return Y, X

def MeshGrid_RPolar(field, gridsize):

    Y, X = MeshGrid(field, gridsize)
    return X**2+Y**2

def Convert_forward(field, gridsize, curvature, wavelength):

    doub1 = curvature
    size = gridsize
    lam = wavelength
    N =field.shape[-1]
    if doub1 == 0.:
        return field
    
    f = -1./doub1
    _2pi = 2*_np.pi
    k = _2pi/lam
    kf = k/(2*f)
    
    RR = MeshGrid_RPolar(field, gridsize)
    Fi = kf*RR
    field *= torch.exp(1j* Fi)
    
    return field, size, curvature

def Convert_inverse(field, gridsize, curvature, wavelength):

    doub1 = curvature
    size = gridsize
    lam = wavelength
    N =field.shape[-1]
    if doub1 == 0.:
        return field
    
    f = -1./doub1
    _2pi = 2*_np.pi
    k = _2pi/lam
    kf = k/(2*f)
    
    RR = MeshGrid_RPolar(field, gridsize)
    Fi = kf*RR
    field /= torch.exp(1j* Fi)
    
    return field, size, curvature

class Spherer2Cartesian(nn.Module):
    def __init__(self, wavelength):
        super(Spherer2Cartesian, self).__init__()
        self.wavelength = wavelength

    def forward(self, Fin, gridsize, curvature):
        Fout, size, curvature = Convert_forward(Fin, gridsize, curvature, self.wavelength)
        
        return Fout, size, curvature
    
class Cartesian2Spherer(nn.Module):
    def __init__(self, wavelength):
        super(Cartesian2Spherer, self).__init__()
        self.wavelength = wavelength

    def forward(self, Fin, gridsize, curvature):
        Fout, size, curvature = Convert_inverse(Fin, gridsize, curvature, self.wavelength)
        
        return Fout, size, curvature
        