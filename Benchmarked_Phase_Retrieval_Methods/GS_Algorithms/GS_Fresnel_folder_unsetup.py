'''
2024.6.18, Shengyuan Yan, TU/e, Eindhoven, NL.
'''

import matplotlib.pyplot as plt
import numpy as np
from LightPipes import *
import os

print(LPversion)
#Parameters used for the experiment:
size=5*mm; #The CCD-sensor has an area of size x size (NB LightPipes needs square grids!)
wavelength=1000.0*nm; #wavelength of the HeNe laser used
z=0.6*m; #propagation distance from near to far field

#Read near and far field (at a distance of z=2 m) from disk:
Inear = np.load('Unet-npy/I.npy')

N=len(Inear)
print(N)

path="GaussZernike-npy-64/val/I/"
Is=os.listdir(path)

for i in range(len(Is)):
    Ifar=np.load(path+Is[i])
    
    F=Begin(size,wavelength,N)

    #The iteration:
    for k in range(1,1000):
        F=SubIntensity(Inear,F) #Substitute the measured far field into the field
        F=Forvard(F, z, usepyFFTW=True) #Propagate back to the near field
        F=SubIntensity(Ifar,F) #Substitute the measured near field into the field
        F=Forvard(F, -z, usepyFFTW=True) #Propagate to the far field
        Phi_pred = Phase(F)
    plt.imsave('GS_pred/img/'+Is[i]+'.png', Phi_pred, cmap='gray')
    np.save('GS_pred/npy/'+Is[i], Phi_pred)
    print(i)