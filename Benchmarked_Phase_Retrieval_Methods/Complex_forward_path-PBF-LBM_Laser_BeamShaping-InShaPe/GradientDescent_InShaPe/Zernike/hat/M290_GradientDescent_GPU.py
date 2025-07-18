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
parser.add_argument('-b', '--batch_size', default=1, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--beamshape', default='RecTophat', type=str)
parser.add_argument('--caustic_plane', default='prefoc', type=str)
parser.add_argument('--optimizer', default='adam', type=str,
                    help='Optimizer algorithm - "adam" for Adam optimizer, "sgd" for SGD optimizer.')
parser.add_argument('--lr', '--learning_rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--step_size', default=50, type=int,
                    help='step size (default: 50)')


iteration = 200

if __name__ == '__main__':
    
    args = parser.parse_args()
    device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")

    beamshape = args.beamshape
    plane = args.caustic_plane
    batchsize = args.batch_size

    vis_dir = 'prediction_result/'

    train_dataset = get_training_data('./')

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batchsize, shuffle=False,
        num_workers=1, pin_memory=True, sampler=None)

    machine = M290(beamshape, 'lightsource.npy', device, plane)
    machine = machine.to(device)

    near_field = machine.nearField
    lightsource = Intensity(near_field)
    plt.imsave('lightsource.png', lightsource.cpu().numpy(), cmap='gray')
    SLM_phase_mask = Phase(near_field)
    plt.imsave('SLMphasemask.png', SLM_phase_mask.cpu().numpy(), cmap='gray')

    near_field = torch.stack([near_field] * batchsize, dim=0)
    lightsource = torch.stack([lightsource] * batchsize, dim=0)
    
    criterion = nn.MSELoss().to(device)
    if args.lr > 0:
        lr = args.lr
    
    
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(machine.parameters(), lr=lr,
                                     weight_decay=1e-4)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(machine.parameters(), lr=lr, momentum=0.9,
                                    weight_decay=1e-4)
    if args.step_size > 0:
        step_size = args.step_size
    scheduler = StepLR(optimizer, step_size=step_size, gamma=0.5)
    
    runtime_list = []
    for i, (beamshape, Z, name) in enumerate(train_loader):

        beamshape = beamshape.to(device)
        
        with torch.no_grad():
            machine.zernike_coeffs.zero_()
        
        start = time.perf_counter()
        for j in range(iteration):

            far_field, size, curvature, phase = machine(near_field, 0, 'M290 machine Forward simulation')
            
            I = torch.abs(far_field)**2
            I = I/torch.max(I)
            I = I*255
            
            loss = criterion(I, beamshape)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            print("Sample "+str(i)+" Iteration "+str(j))
            print(loss.item())
            
        end = time.perf_counter()
        runtime_list.append(end-start)
        print(f"Runtime: {end - start:.4f} seconds")
        
        I = I.cpu().detach().numpy()
        mpimg.imsave(vis_dir+name[0][:7]+'_intensity.png', -I[0], cmap='Greys')
        np.save(vis_dir+"/pred_I/"+name[0][:7]+'_intensity.npy', I[0])
        
        phase = phase.cpu().detach().numpy()
        mpimg.imsave(vis_dir+name[0][:7]+'_phase.png', phase[0], cmap='Greys')
        np.save(vis_dir+"/pred_phi/"+name[0][:7]+'_phase.npy', phase[0])
        
        z_pred = machine.zernike_coeffs
        z_pred = z_pred.cpu().detach().numpy()
        np.save(vis_dir+"/pred_z/"+name[0][:7]+'_zernike.npy', z_pred)
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    runtime_list = np.array(runtime_list)
    np.save("runtime.npy", runtime_list)