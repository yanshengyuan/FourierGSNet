import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from M290_MachineSimu_GPU.M290 import M290
from M290_MachineSimu_GPU.complex_field_tools_GPU.complex_field_tools import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

beamshape = 'Chair'
plane = 'prefoc'

vis_dir = 'pics/'

Zernike_Coeff1 = np.load('Output_Data/run0000_zernikeCoeff.npy')[3:]
Zernike_Coeff2 = np.load('Output_Data/run0001_zernikeCoeff.npy')[3:]
Zernike_Coeff1 = torch.from_numpy(Zernike_Coeff1).to(device)
Zernike_Coeff2 = torch.from_numpy(Zernike_Coeff2).to(device)

Zernike_Coeffs = torch.stack([Zernike_Coeff1, Zernike_Coeff2], dim=0)
fake_Zernike_Coeffs = torch.zeros_like(Zernike_Coeffs, device = device)

machine = M290(beamshape, 'lightsource.npy', device, plane)

near_field = machine.nearField

lightsource = Intensity(near_field)
plt.imsave('lightsource.png', lightsource.cpu().numpy(), cmap='gray')

SLM_phase_mask = Phase(near_field)
plt.imsave('SLMphasemask.png', SLM_phase_mask.cpu().numpy(), cmap='gray')

batchsize = Zernike_Coeffs.shape[0]
near_field = torch.stack([near_field] * batchsize, dim=0)
lightsource = torch.stack([lightsource] * batchsize, dim=0)

far_field, size, curvature = machine.M290_forward(near_field, Zernike_Coeffs)

far_phase = Phase(far_field)
far_phase = far_phase.cpu().numpy()
far_intensity = Intensity(far_field, flag=2)
far_intensity = far_intensity.cpu().numpy()

for i in range(len(far_phase)):
    plt.imsave(vis_dir+'prop_phase_'+str(i+1)+'.png', far_phase[i], cmap='gray')
    plt.imsave(vis_dir+'prop_intensity_'+str(i+1)+'.png', far_intensity[i], cmap='gray')
    np.save("../compare_results/prop_intensity_gpu_"+str(i+1)+".npy", far_intensity[i])
    np.save("../compare_results/prop_phase_gpu_"+str(i+1)+".npy", far_phase[i])