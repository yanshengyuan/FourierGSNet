'''
2024.10.28, Shengyuan Yan, TU/e, Eindhoven, NL.
'''

import matplotlib.pyplot as plt
import numpy as np
import os
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


Inear = np.load('Unet-npy/I-phaseimaging.npy').astype(np.float32)
Inear=torch.from_numpy(Inear)
AmpNear=torch.sqrt(Inear)
AmpNear=AmpNear.to(device)

path="tiny-imagenet-gray/I_test/npy/"
Is=os.listdir(path)

phase_init=np.zeros_like(Inear).astype(np.float32)
phase_init=torch.from_numpy(phase_init)
phase_init=phase_init.to(device)

for i in range(len(Is)):
    Ifar=np.load(path+Is[i]).astype(np.float32)
    Ifar=torch.from_numpy(Ifar)
    AmpFar=torch.sqrt(Ifar)
    AmpFar=AmpFar.to(device)
    
    F_near=AmpNear*torch.exp(phase_init * 1j)

    #The iteration:
    for k in range(1,2000):
        
        F_far = torch.fft.fft2(F_near)
        phase_far = torch.angle(F_far)
        F_far = AmpFar*torch.exp(phase_far * 1j)
        
        F_near = torch.fft.ifft2(F_far) 
        phase_near = torch.angle(F_near)
        F_near = AmpNear*torch.exp(phase_near * 1j)
        
    F_far = torch.fft.fft2(F_near)
    F_far=F_far.to('cpu')
    phase_near=phase_near.to('cpu')
        
    plt.imsave('GS_FFT_pred/img/'+Is[i]+'.png', phase_near, cmap='gray')
    plt.imsave('GS_FFT_pred/I/'+Is[i]+'.png', np.absolute(F_far)**2, cmap='gray')
    np.save('GS_FFT_pred/npy/'+Is[i], phase_near)
    print(i)