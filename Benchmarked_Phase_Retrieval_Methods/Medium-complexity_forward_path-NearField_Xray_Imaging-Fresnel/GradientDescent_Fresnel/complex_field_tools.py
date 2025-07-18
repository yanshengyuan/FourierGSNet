"""
Shengyuan Yan, TU/e, Eindhoven, Netherlands, 01:57am, 4/08/2025
"""

import numpy as np
import torch
from torch import nn

def Wrap(phase_tensor):
    
    return torch.fmod(phase_tensor + torch.pi, 2 * torch.pi) - torch.pi

def Phase(field):
    
    if isinstance(field, torch.Tensor):
        Phi = torch.angle(field)
    
    if isinstance(field, np.ndarray):
        Phi = np.angle(field)
    
    return Phi

def Intensity(field, flag=0):
    
    if isinstance(field, torch.Tensor):
        I = torch.abs(field)**2
    
    if isinstance(field, np.ndarray):
        I = np.abs(field)**2
    
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
    
    if isinstance(field, torch.Tensor):
        if isinstance(Intens, torch.Tensor):
            phi = torch.angle(field)
            Efield = torch.sqrt(Intens)
            field = Efield * torch.exp(1j * phi)
    
    if isinstance(field, np.ndarray):
        if isinstance(Intens, np.ndarray):
            phi = np.angle(field)
            Efield = np.sqrt(Intens)
            field = Efield * np.exp(1j * phi)
    
    return field

def SubPhase(field, Phi):
    
    if Phi.shape != field.shape:
        raise ValueError('Phase map has wrong shape')
    
    if isinstance(field, torch.Tensor):
        if isinstance(Phi, torch.Tensor):
            oldabs = torch.abs(field)
            field = oldabs * torch.exp(1j * Phi)
            
    if isinstance(field, np.ndarray):
        if isinstance(Phi, np.ndarray):
            oldabs = np.abs(field)
            field = oldabs * np.exp(1j * Phi)
    
    return field

def Inv_Squares(xn,yn, field, dx):
    
    N = field.shape[0]
    No2 = int(N/2)
    II = np.floor(xn/dx+No2).astype(int)
    JJ = np.floor(yn/dx+No2).astype(int)
    
    x = (II-No2)*dx
    y = (JJ-No2)*dx
    
    tol = 1e-6*dx
    if (np.any(xn < x-tol) or np.any(xn > x+dx+tol) or
        np.any(yn<y-tol) or np.any(yn > y+dx+tol)):
        raise ValueError('Out of range')
    
    xlow = xn-x
    xhigh = x+dx-xn
    ylow = yn-y
    yhigh = y+dx-yn
    
    if (np.any(xlow < -tol) or np.any(xhigh < -tol) or
        np.any(ylow < -tol) or np.any(yhigh < -tol)):
        raise ValueError('Out of range')

    z = field[JJ, II]
    zx = field[JJ, II+1]
    zy = field[JJ+1, II]
    zxy = field[JJ+1, II+1]
    
    zout = yhigh * (z*xhigh + zx*xlow)
    zout += ylow * (zy*xhigh + zxy*xlow)
    zout /= dx**2
    return zout

def field_interpolate(Fin, old_size, new_size, old_N, new_N, x_shift = 0.0, y_shift = 0.0, angle = 0.0, magnif = 1.0 ):

    Fout = np.zeros((new_N, new_N), dtype=np.complex64)
    
    Pi = 3.141592654
    
    angle *= Pi/180.
    cc=np.cos(angle)
    ss=np.sin(angle)
    
    size_old = old_size
    old_number = old_N
    dx_old = size_old/(old_number-1)
    on21 = int(old_number/2)
    Xold = dx_old * np.arange(-on21, old_number-on21)
    Yold = dx_old * np.arange(-on21, old_number-on21)
    
    dx_new = new_size/(new_N-1)
    nn21 = int(new_N/2)
    X0 = dx_new * np.arange(-nn21, new_N-nn21)
    Y0 = dx_new * np.arange(-nn21, new_N-nn21)
    X0, Y0 = np.meshgrid(X0, Y0)
    
    X0 -= x_shift
    Y0 -= y_shift
    Xnew = (X0*cc + Y0*ss)/magnif
    Ynew = (X0*(-ss) + Y0* cc)/magnif
    
    xmin, xmax = Xold[0], Xold[-1]
    ymin, ymax = Yold[0], Yold[-1]

    filtmask = ((Xnew > xmin) & (Xnew < xmax) &
                (Ynew > ymin) & (Ynew < ymax))

    Xmask = Xnew[filtmask]
    Ymask = Ynew[filtmask]
    
    out_z = Inv_Squares(Xmask, Ymask, Fin, dx_old)
    Fout[filtmask] = out_z
    
    Fout /= magnif

    return Fout