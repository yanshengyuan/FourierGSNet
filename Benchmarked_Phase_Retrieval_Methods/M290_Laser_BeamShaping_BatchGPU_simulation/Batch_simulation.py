import argparse
import os
import shutil
import time
import warnings
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset
from utils.loader import get_training_data

from M290_MachineSimu_GPU.M290 import M290
from M290_MachineSimu_GPU.complex_field_tools_GPU.complex_field_tools import *

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('-b', '--batch_size', default=5, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--beamshape', default='RecTophat', type=str)
parser.add_argument('--caustic_plane', default='prefoc', type=str)

start = time.perf_counter()

if __name__ == '__main__':
    
    args = parser.parse_args()
    device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")

    beamshape = args.beamshape
    plane = args.caustic_plane
    batchsize = args.batch_size

    vis_dir = 'Simulation_result/'

    train_dataset = get_training_data('./')

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batchsize, shuffle=False,
        num_workers=4, pin_memory=True, sampler=None)

    first_batch, _ = next(iter(train_loader))
    fake_Zernike_Coeffs = torch.zeros_like(first_batch, device = device)

    machine = M290(beamshape, 'lightsource.npy', device, plane)

    near_field = machine.nearField
    lightsource = Intensity(near_field)
    plt.imsave('lightsource.png', lightsource.cpu().numpy(), cmap='gray')
    SLM_phase_mask = Phase(near_field)
    plt.imsave('SLMphasemask.png', SLM_phase_mask.cpu().numpy(), cmap='gray')

    near_field = torch.stack([near_field] * batchsize, dim=0)
    lightsource = torch.stack([lightsource] * batchsize, dim=0)
    
    for i, (Z, name) in enumerate(train_loader):
        
        Zernike_Coeffs = Z.to(device, non_blocking=True).squeeze()
        
        far_field, size, curvature = machine(near_field, 0, 'M290 machine Forward simulation', Zernike_Coeffs)

        #far_phase = Phase(far_field)
        #far_phase = far_phase.cpu().numpy()
        far_intensity = Intensity(far_field, flag=2)
        far_intensity = far_intensity.cpu().numpy()

        for j in range(len(far_intensity)):
            mpimg.imsave(vis_dir+name[j][:7]+'_intensity.png', -far_intensity[j], cmap='Greys')
        
        print("Batch "+str(i))

end = time.perf_counter()
print(f"Simulation time: {end - start:.4f} seconds")