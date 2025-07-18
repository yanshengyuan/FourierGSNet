"""
Shengyuan Yan, TU/e, Eindhoven, Netherlands, 17:00pm, 4/02/2025
"""

import torch
from torch import nn

def Phase(field):

    Phi = torch.angle(field)
    
    return Phi

def Intensity(field, flag=0):
    
    I = torch.abs(field)**2
    
    if flag > 0:
        for i in range(len(I)):
            Imax = I[i].max()
            if Imax == 0.0:
                raise ValueError('Cannot normalize because of 0 beam power.')
            I[i] = I[i] / Imax
        
        if flag == 2:
            I = I * 255
            
    return I

def SubIntensity(field, Intens):
    
    if Intens.shape != field.shape:
        raise ValueError('Intensity map has wrong shape')
    
    phi = torch.angle(field)
    Efield = torch.sqrt(Intens)
    
    field = Efield * torch.exp(1j * phi)
    
    return field

def SubPhase(field, Phi):
    
    if Phi.shape != field.shape:
        raise ValueError('Phase map has wrong shape')
    
    oldabs = torch.abs(field)
    field = oldabs * torch.exp(1j * Phi)
    
    return field