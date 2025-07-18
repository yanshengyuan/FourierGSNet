"""
Shengyuan Yan, TU/e, Eindhoven, Netherlands, 21:00pm, 4/02/2025
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import time

from M290_MachineSimu_GPU.M290 import M290
from M290_MachineSimu_GPU.complex_field_tools_GPU.complex_field_tools import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

beamshape = 'ring'
plane = 'prefoc'

vis_dir = 'M290_GS_algorithm_results/'

phase1 = np.load('exmp_data/'+beamshape+'_phase_'+plane+'_1.npy')
plt.imsave(vis_dir+'phase_start_1.png', phase1, cmap='gray')
phase1 = torch.from_numpy(phase1).to(device)
intensity1 = np.load('exmp_data/'+beamshape+'_intensity_'+plane+'_1.npy')
plt.imsave(vis_dir+'beamshape_1.png', intensity1, cmap='gray')

phase2 = np.load('exmp_data/'+beamshape+'_phase_'+plane+'_2.npy')
plt.imsave(vis_dir+'phase_start_2.png', phase2, cmap='gray')
phase2 = torch.from_numpy(phase2).to(device)
intensity2 = np.load('exmp_data/'+beamshape+'_intensity_'+plane+'_2.npy')
plt.imsave(vis_dir+'beamshape_2.png', intensity2, cmap='gray')

phase = torch.stack([phase1, phase2], dim=0)

machine = M290('lightsource.npy', device, plane)
lightsource = machine.lightsource
init_field = machine.initField
plt.imsave('lightsource.png', lightsource.cpu().numpy(), cmap='gray')

batchsize = phase.shape[0]
near_field = machine.initField
near_field = torch.stack([near_field] * batchsize, dim=0)
lightsource = torch.stack([lightsource] * batchsize, dim=0)

near_field = SubIntensity(near_field, lightsource)
near_field = SubPhase(near_field, phase)

far_field, size, curvature = machine.M290_forward(near_field)

far_phase = Phase(far_field)
far_phase = far_phase.cpu().numpy()
far_intensity = Intensity(far_field, flag=2)
far_intensity = far_intensity.cpu().numpy()

for i in range(len(far_phase)):
    plt.imsave(vis_dir+'prop_phase_'+str(i+1)+'.png', far_phase[i], cmap='gray')
    plt.imsave(vis_dir+'prop_intensity_'+str(i+1)+'.png', far_intensity[i], cmap='gray')

inv_field = machine.M290_inverse(far_field, size, curvature)

inv_phase = Phase(inv_field)
inv_lightsource = Intensity(inv_field, flag=2)

phase_check = inv_phase
phase_check = phase_check.cpu().numpy()
#np.save("../compare_results/inv_phase_gpu.npy", phase_check)

intensity_check = Intensity(inv_field, flag=2)
intensity_check = intensity_check.cpu().numpy()

for i in range(len(phase_check)):
    plt.imsave(vis_dir+'inv_phase_'+str(i+1)+'.png', phase_check[i], cmap='gray')
    plt.imsave(vis_dir+'inv_lightsource_'+str(i+1)+'.png', intensity_check[i], cmap='gray')

inv_field = SubPhase(near_field, inv_phase)
inv_field = SubIntensity(inv_field, lightsource)

prop_inv_field, size, curvature = machine.M290_forward(inv_field)

prop_inv_phase = Phase(prop_inv_field)
prop_inv_phase = prop_inv_phase.cpu().numpy()
prop_inv_intensity = Intensity(prop_inv_field, flag=2)
prop_inv_intensity = prop_inv_intensity.cpu().numpy()

for i in range(len(prop_inv_phase)):
    plt.imsave(vis_dir+'prop_inv_phase_'+str(i+1)+'.png', prop_inv_phase[i], cmap='gray')
    plt.imsave(vis_dir+'prop_inv_intensity_'+str(i+1)+'.png', prop_inv_intensity[i], cmap='gray')

#np.save("../compare_results/prop_inv_phase_gpu.npy", prop_inv_intensity)

machine = M290('lightsource.npy', device, plane)
init_field = machine.initField
init_field = torch.stack([init_field] * batchsize, dim=0)

intensity1 = torch.from_numpy(intensity1).to(device)
intensity2 = torch.from_numpy(intensity2).to(device)
beamshapes = torch.stack([intensity1, intensity2], dim=0)
field = init_field

'''
random_init = (torch.rand((batchsize, field.shape[-1], field.shape[-1]), device=device))*2*torch.pi-torch.pi
field = SubPhase(field, random_init)
'''

iterations = 100

start=time.perf_counter()

for i in range(iterations):
    field = SubIntensity(field, lightsource)
    field, size, curvature = machine.M290_forward(field)
    
    #'''
    GS_intensity = Intensity(field)
    GS_intensity = GS_intensity.cpu().numpy()
    
    fig, axes = plt.subplots(2, batchsize, figsize=(5 * batchsize, 5 * batchsize))
    for j in range(len(GS_intensity)):
        axes[j][0].imshow(GS_intensity[j])
        axes[j][0].axis("off")
    #'''
    
    if(i==iterations-1):
        GS_intensity = Intensity(field, flag=2)
        GS_intensity = GS_intensity.cpu().numpy()
        for j in range(len(GS_intensity)):
            plt.imsave(vis_dir+"GS_intensity_"+str(j)+".png", GS_intensity[j], cmap='gray')
    
    field = SubIntensity(field, beamshapes)
    field = machine.M290_inverse(field, size, curvature)
    
    #'''
    GS_phase = Phase(field)
    GS_phase = GS_phase.cpu().numpy()
    
    for j in range(len(GS_phase)):
        axes[j][1].imshow(GS_phase[j], cmap='gray')
        axes[j][1].axis("off")
            
    plt.show()
    plt.close()
    #'''
    
    if(i==iterations-1):
        GS_phase = Phase(field)
        GS_phase = GS_phase.cpu().numpy()
        for j in range(len(GS_phase)):
            plt.imsave(vis_dir+"GS_phase_"+str(j)+".png", GS_phase[j], cmap='gray')
    
    print(i)
    
end=time.perf_counter()
t=(end-start)
print(f"Runtime: {t:.3f} seconds")