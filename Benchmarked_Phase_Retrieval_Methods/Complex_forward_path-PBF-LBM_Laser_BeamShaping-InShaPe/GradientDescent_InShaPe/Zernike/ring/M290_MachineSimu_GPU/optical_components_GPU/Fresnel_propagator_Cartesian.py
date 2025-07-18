"""
Shengyuan Yan, TU/e, Eindhoven, Netherlands, 17:00pm, 4/02/2025
"""

import numpy as np
import torch
from torch import nn
from torch.fft import fft2 as _fft2
from torch.fft import ifft2 as _ifft2

def Fresnel_ASM_GPUandINV(Fin, gridsize, wavelength, z):
    
    device = Fin.device
    
    _fftargs = {}
    if z==0:
        return Fin
    
    size = gridsize
    lam = wavelength
    
    N = Fin.size(-1)
    
    _2pi = 2*np.pi
    
    if (z==0):
        return Fin
    zz = z
    z = abs(z)
    kz = _2pi/lam*z
    cokz = np.cos(kz)
    sikz = np.sin(kz)
    
    iiN = np.ones((N,),dtype=float)
    iiN[1::2] = -1
    iiij = np.outer(iiN, iiN)
    iiij = torch.from_numpy(iiij)
    if Fin.is_cuda:
        iiij = iiij.to(device)
    Fin *= iiij
    
    z1 = z*lam/2
    No2 = int(N/2)

    SW = np.arange(-No2, N-No2)/size
    SW *= SW
    SSW = SW.reshape((-1,1)) + SW
    Bus = z1 * SSW
    Ir = Bus.astype(int)
    Abus = _2pi*(Ir-Bus)
    Cab = np.cos(Abus)
    Sab = np.sin(Abus)
    CC = Cab + 1j * Sab
    CC=torch.from_numpy(CC)
    if Fin.is_cuda:
        CC = CC.to(device)
    
    if zz >= 0.0:
        Fin = _fft2(Fin, **_fftargs)
        Fin *= CC
        Fin = _ifft2(Fin, **_fftargs)
    else:
        Fin = _ifft2(Fin, **_fftargs)
        CCB = CC.conj()
        Fin *= CCB
        Fin = _fft2(Fin, **_fftargs)
    
    Fin *= (cokz + 1j* sikz)
    Fin *= iiij
    
    return Fin

class Fresnel_propagator(nn.Module):
    def __init__(self, wavelength):
        super(Fresnel_propagator, self).__init__()
        self.wavelength = wavelength

    def forward(self, Fin, z, gridsize):
        
        Fout = Fresnel_ASM_GPUandINV(Fin, gridsize, self.wavelength, z)

        return Fout