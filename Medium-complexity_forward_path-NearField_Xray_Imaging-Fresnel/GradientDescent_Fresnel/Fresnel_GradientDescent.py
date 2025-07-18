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

from Dataset2channel import *
from model_ICLR import GSNet_Fresnel as PImodel
from Fresnel_propagator import Fresnel_propagator
from complex_field_tools import *

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('-b', '--batch_size', default=1, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--optimizer', default='adam', type=str,
                    help='Optimizer algorithm - "adam" for Adam optimizer, "sgd" for SGD optimizer.')
parser.add_argument('--lr', '--learning_rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--step_size', default=50, type=int,
                    help='step size')
parser.add_argument('--data', metavar='DIR', nargs='?', default='imagenet',
                    help='path to dataset (default: imagenet)')


iteration = 200

if __name__ == '__main__':
    
    args = parser.parse_args()
    device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")

    batchsize = args.batch_size

    vis_dir = 'prediction_result/'

    val_dataset = Dataset2channel(args.data + '/test', recursive=False, load_data=False,
                                   data_cache_size=1, transform=transforms.ToTensor())
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                              batch_size=args.batch_size, shuffle=False)
    first_I, _, _ = next(iter(val_loader))
    HW = first_I.shape[-1]

    machine = Fresnel_propagator(HW)
    machine = machine.to(device)
    
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
    for i, (I, real, imag) in enumerate(val_loader):
        
        Phi=torch.atan2(imag, real)
        I_near = real**2 + imag**2
        phi_min = Phi.min()
        phi_max = Phi.max()

        Phi = Phi.cuda(args.gpu, non_blocking=True).squeeze()
        I_near = I_near.cuda(args.gpu, non_blocking=True).squeeze()
        I_far = I.cuda(args.gpu, non_blocking=True).squeeze()
        
        '''
        machine.phase_object.data.copy_(Phi)
        field_check = machine(I_near)
        I = torch.abs(field_check)**2
        I = I/torch.max(I)
        I = I*255
        I = I.cpu().detach().numpy()
        mpimg.imsave(str(i)+'_check.png', -I, cmap='Greys')
        '''
        
        phase_randinit = (phi_max-phi_min)*torch.rand((HW, HW), dtype=torch.float32)+phi_min
        
        #'''
        with torch.no_grad():
            machine.phase_object.zero_()
        #'''
        
        #machine.phase_object.data.copy_(phase_randinit)

        start = time.perf_counter()
        for j in range(iteration):

            far_field = machine(I_near)
            
            I = torch.abs(far_field)**2
            
            I_far = I_far.squeeze()
            loss = criterion(I, I_far)
            
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            scheduler.step()
            
            print("Sample "+str(i)+" Iteration "+str(j))
            print(loss.item())
            
        end = time.perf_counter()
        runtime_list.append(end-start)
        print(f"Runtime: {end - start:.4f} seconds")
        
        I = I/torch.max(I)
        I = I*255
        I = I.cpu().detach().numpy()
        mpimg.imsave(vis_dir+str(i)+'_intensity.png', -I, cmap='Greys')
        np.save(vis_dir+"/pred_I/"+str(i)+'_intensity.npy', I)
        
        Phi = Phi.cpu().detach().numpy()
        np.save(vis_dir+"/gt_phi/"+str(i)+'_phase_gt.npy', Phi)
        
        phase = machine.phase_object.detach()
        phase = Wrap(phase)
        phase = phase.cpu().numpy()
        mpimg.imsave(vis_dir+str(i)+'_phase.png', phase, cmap='Greys')
        np.save(vis_dir+"/pred_phi/"+str(i)+'_phase.npy', phase)
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    runtime_list = np.array(runtime_list)
    np.save("runtime.npy", runtime_list)