"""
Shengyuan Yan, TU/e, Eindhoven, Netherlands, 18:00pm, 4/08/2025
"""

import numpy as _np
import torch
from torch import nn

def zernike(n,m,rho,phi):
    
    mabs = _np.abs(m)
    prod = 1.0
    sign = 1
    summ = 0.0
    for s in range(int((n-mabs)/2) + 1):
        if n-2*s != 0:
            prod = _np.power(rho, n-2*s)
        else:
            prod = 1.0
        prod *= _np.math.factorial(n-s)*sign
        prod /= (_np.math.factorial(s)
                * _np.math.factorial(int(((n+mabs)/2))-s)
                * _np.math.factorial(int(((n-mabs)/2))-s))
        summ += prod
        sign = -sign
    if m>=0:
        return summ*_np.cos(m*phi)
    else:
        return (-1)*summ*_np.sin(m*phi)

def Zernike_Polynomial(wavelength, gridsize, polar_radius, polar_angle, n, m, R,
                       A = 1.0, norm=True, units='rad'):

    mcorrect = False
    ncheck = n
    while ncheck >= -n:
        if ncheck == m:
            mcorrect = True
        ncheck -= 2
    if not mcorrect:
        raise ValueError('Zernike: n,m must fulfill: n>0, |m|<=n and n-|m|=even')
    
    k = 2*_np.pi/wavelength
    
    if units=='opd':
        A = k*A
    elif units=='lam':
        A = 2*_np.pi*A
    elif units=='rad':
        A = A
    else:
        raise ValueError('Unknown value for option units={}'.format(units))
    
    if norm:
        if m==0:
            Nnm = _np.sqrt(n+1)
        else:
            Nnm = _np.sqrt(2*(n+1))
    else:
        Nnm = 1
    
    rho = polar_radius/R
    fi = -A*Nnm*zernike(n,m, rho, polar_angle)
    
    return fi

def noll_to_zern(j):

    if (j == 0):
        raise ValueError("Noll indices start at 1, 0 is invalid.")

    n = 0
    j1 = j-1
    while (j1 > n):
        n += 1
        j1 -= n

    m = (-1)**j * ((n % 2) + 2 * int((j1+((n+1)%2)) / 2.0 ))
    return (n, m)