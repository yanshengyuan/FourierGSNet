"""
Shengyuan Yan, TU/e, Eindhoven, Netherlands, 17:00pm, 4/02/2025
"""

from M290_MachineSimu_GPU.optical_components_GPU.Fresnel_propagator_Cartesian import Fresnel_propagator
from M290_MachineSimu_GPU.complex_field_tools_GPU.complex_field_tools import *

import numpy as _np
import torch
from torch import nn

def Fresnel_VarSpherer_forward(propagator, field, f, z, gridsize, curvature, wavelength):

    LARGENUMBER = 10000000.
    doub1 = curvature
    size = gridsize
    lam = wavelength
    
    if doub1 !=0.:
        f1 = 1/doub1
    else:
        f1 = LARGENUMBER * size**2/lam
        
    if (f+f1) != 0.:
        f = (f*f1)/(f+f1)
    else:
        f = LARGENUMBER * size**2/lam
    
    if ((z-f) == 0 ):
        z1 = LARGENUMBER
    else:
        z1= -z*f/(z-f)
    
    field = propagator(field, z1, size)
    
    ampl_scale = (f-z)/f
    size *= ampl_scale
    doub1 = -1./(z-f)
    curvature = doub1

    field /= ampl_scale

    return field, size, curvature, ampl_scale, z1

def Fresnel_VarSpherer_inverse(propagator, field, z1, size, curvature, ampl_scale):

    field *= ampl_scale
    size /= ampl_scale
    curvature = 0
    
    field = propagator(field, z1, size)

    return field, size, curvature


class Expander_Fresnel(nn.Module):
    def __init__(self, wavelength):
        super(Expander_Fresnel, self).__init__()
        self.wavelength = wavelength
        self.propagator = Fresnel_propagator(self.wavelength)
        self.amp_scale = 0
        self.z1 = 0
        self.size = 0
        self.curvature = 0

    def forward(self, Fin, f, z, gridsize, curvature):
        if(z>0):
            Fout = Fresnel_VarSpherer_forward(self.propagator, Fin, f, z, gridsize, curvature, self.wavelength)
            field, self.size, self.curvature, self.amp_scale, self.z1 = Fout
            
            return field, self.size, self.curvature
            
        else:
            Fout = Fresnel_VarSpherer_inverse(self.propagator, Fin, -self.z1, self.size, self.curvature, self.amp_scale)
            
            return Fout