# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 10:23:57 2022

@author: aga
"""

import numpy as np
from scipy.special import comb
from LightPipes.field import Field


def SmoothStep(x, x_min=0, x_max=1, N=0):
    """
    * Creates a step function with a smooth step *
    see https://stackoverflow.com/questions/45165452/how-to-implement-a-smooth-clamp-function-in-python
    Used to create circular aperture with smooth edges.
    
    :param x: evaluation points x of the function
    :type x: int, float numpy array
    :param x_min number: lower bound of edge
    :type x_min: int, float
    :param x_max number: upper bound of edge
    :type x_max: int, float
    :param N: order of edge polynomial is N+1
    :type N: int
    
    """

    x = np.clip((x - x_min) / (x_max - x_min), 0, 1)

    result = 0
    for n in range(0, N + 1):
         result += comb(N + n, n) * comb(2 * N + 1, N - n) * (-x) ** n

    result *= x ** (N + 1)

    return result


def SmoothCircAperture(Fin, R, s, x_shift = 0.0, y_shift = 0.0, n = 2, T = 1.0):
    """
    *Inserts an aperture with a smooth edges in the field.*
    
    :param Fin: input field
    :type Fin: Field
    :param R: Aperture radius
    :type R: int, float
    :param s: Aperture edge width
    :type s: int, float
    :param x_shift: shift in x direction (default = 0.0)
    :param y_shift: shift in y direction (default = 0.0)
    :type x_shift: int, float
    :type y_shift: int, float
    :param n: order of edge polynomial is n+1 (default = 2)
    :type n: int 
    :param T: center intensity transmission (default = 1.0)
    :type T: int, float
    :return: output field (N x N square array of complex numbers).
    :rtype: `LightPipes.field.Field`
  
    """ 
    Fout = Field.copy(Fin)
    
    Y, X = Fout.mgrid_cartesian
    Y = Y - y_shift
    X = X - x_shift

    SqrtT=np.sqrt(T)
    
    # Use SmootStep function here
    Fout.field*=SqrtT * np.sqrt(SmoothStep(R-np.sqrt(X*X+Y*Y),-s/2,s/2,n))
    Fout._IsGauss=False
    return Fout


