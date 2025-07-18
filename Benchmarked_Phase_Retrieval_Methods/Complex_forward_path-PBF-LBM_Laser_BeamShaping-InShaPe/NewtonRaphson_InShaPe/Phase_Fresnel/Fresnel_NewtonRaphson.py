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
parser.add_argument('--data', metavar='DIR', nargs='?', default='imagenet',
                    help='path to dataset (default: imagenet)')

def poisson_loss(M_pred, I_obs, eps=1e-3):
    
    poisson_log_likelihood = (M_pred - I_obs * torch.log(M_pred + eps)).sum()
    
    return poisson_log_likelihood

iteration = 30

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
    
    runtime_list = []
    for i, (I, real, imag) in enumerate(val_loader):
        
        Phi=torch.atan2(imag, real)
        I_near = real**2 + imag**2
        phi_min = Phi.min()
        phi_max = Phi.max()

        Phi = Phi.cuda(args.gpu, non_blocking=True).squeeze()
        I_near = I_near.cuda(args.gpu, non_blocking=True).squeeze()
        I_far = I.cuda(args.gpu, non_blocking=True).squeeze()
        
        phase_randinit = (phi_max-phi_min)*torch.rand((HW, HW), dtype=torch.float32)+phi_min
        
        #'''
        with torch.no_grad():
            machine.phase_object.zero_()
        #'''
        
        #machine.phase_object.data.copy_(phase_randinit)
        
        num_pixels = machine.phase_object.shape[-1]
        start = time.perf_counter()
        for j in range(iteration):

            far_field = machine(I_near)
            
            I = torch.abs(far_field)**2
            
            I_far = I_far.squeeze()
            loss = poisson_loss(I, I_far)
            
            machine.zero_grad() # equals to optimizer.zero_grad()
            grad = torch.autograd.grad(loss, machine.phase_object, create_graph=True)[0].view(-1)
            
            Hessian = []
            for k in range(grad.numel()):
                print(k)
                grad_k = torch.autograd.grad(grad[k], machine.phase_object, retain_graph=True)[0]
                Hessian.append(grad_k.view(-1))
            Hessian = torch.stack(Hessian, dim=0)
            
            Hessian += torch.eye(num_pixels, device=device) * 1e-3
            
            # Newton-Raphson step
            delta = torch.linalg.solve(Hessian, grad)
            
            with torch.no_grad():
                machine.phase_object -= delta
            
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
    
    runtime_list = np.array(runtime_list)
    np.save("runtime.npy", runtime_list)