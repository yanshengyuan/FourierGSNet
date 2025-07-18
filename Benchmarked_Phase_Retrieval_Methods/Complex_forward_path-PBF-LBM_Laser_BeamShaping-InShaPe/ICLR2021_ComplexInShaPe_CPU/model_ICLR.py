import torch
import torch.nn as nn
import torch.nn.functional as f
from einops import rearrange
import numpy as np
import matplotlib.pyplot as plt

from LightPipes import *
from M290_machine_simu.M290 import M290

class Simu_layer(nn.Module):
    def __init__(self, args, machine):
        super(Simu_layer, self).__init__()
        
        self.phase_updating_network = UNet_original()
        self.machine = machine
        
    def forward(self, phase, intens_far, forward_flip=True, inverse_flip=False):
        
        if isinstance(phase, torch.Tensor):
            phase = phase.squeeze()
            phase = phase.detach().to("cpu").numpy()
        near_field = SubPhase(self.machine.nearField, phase)
        
        if(forward_flip==False):
            far_field = self.machine.M290_forward(near_field, flip=False)
        if(forward_flip==True):
            far_field = self.machine.M290_forward(near_field)
        
        F = far_field.field
        real_far = F.real
        real_far = torch.from_numpy(real_far.copy()).cuda().float()
        real_far = real_far.unsqueeze(0)
        real_far = real_far.unsqueeze(0)
        imag_far = F.imag
        imag_far = torch.from_numpy(imag_far.copy()).cuda().float()
        imag_far = imag_far.unsqueeze(0)
        imag_far = imag_far.unsqueeze(0)
        
        beamshape = intens_far.detach().to("cpu").numpy()
        far_field = SubIntensity(far_field, beamshape)
        
        if(inverse_flip==True):
            near_field = self.machine.M290_inverse(far_field, flip=True)
        if(inverse_flip==False):
            near_field = self.machine.M290_inverse(far_field)
        
        phase = Phase(near_field)
        phase = torch.from_numpy(phase).cuda().float()
        phase = phase.unsqueeze(0)
        phase = phase.unsqueeze(0)
        intens_far = intens_far.unsqueeze(0)
        intens_far = intens_far.unsqueeze(0)
        tensor = torch.stack((real_far, imag_far, intens_far, phase), dim=1)
        tensor = tensor.squeeze(2)
        
        phase = self.phase_updating_network(tensor)
        
        return phase
    
class GSNet_Simu(nn.Module):
    def __init__(self, args, num_layers=11):
        super(GSNet_Simu, self).__init__()
        
        self.machine = M290(args.data+"/lightsource.npy")
        self.init_phase = Phase(self.machine.initField)
        
        self.layers = nn.ModuleList()
        self.input_layer = Simu_layer(args, self.machine)
        self.output_layer = Simu_layer(args, self.machine)
        
        for _ in range(num_layers-2):
            layer = Simu_layer(args, self.machine)
            self.layers.append(layer)
        
    def forward(self, intens_far):
        
        phase = self.input_layer(self.init_phase, intens_far, forward_flip=False)
        for layer in self.layers:
            phase = layer(phase, intens_far)
        phase = self.output_layer(phase, intens_far, inverse_flip=True)
        
        return phase
    
#########################################
class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, strides=1):
        super(ConvBlock, self).__init__()
        self.strides = strides
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=strides, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=strides, padding=1),
            nn.LeakyReLU(inplace=True),
        )
        self.conv11 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=strides, padding=0)

    def forward(self, x):
        out1 = self.block(x)
        out2 = self.conv11(x)
        out = out1 + out2
        return out

    
class UNet_original(nn.Module):
    def __init__(self, block=ConvBlock,dim=32):
        super(UNet_original, self).__init__()

        self.dim = dim
        self.ConvBlock1 = ConvBlock(4, dim, strides=1)
        self.pool1 = nn.Conv2d(dim,dim,kernel_size=4, stride=2, padding=1)

        self.ConvBlock2 = block(dim, dim*2, strides=1)
        self.pool2 = nn.Conv2d(dim*2,dim*2,kernel_size=4, stride=2, padding=1)
       
        self.ConvBlock3 = block(dim*2, dim*4, strides=1)
        self.pool3 = nn.Conv2d(dim*4,dim*4,kernel_size=4, stride=2, padding=1)
       
        self.ConvBlock4 = block(dim*4, dim*8, strides=1)
        self.pool4 = nn.Conv2d(dim*8, dim*8,kernel_size=4, stride=2, padding=1)

        self.ConvBlock5 = block(dim*8, dim*16, strides=1)

        self.upv6 = nn.ConvTranspose2d(dim*16, dim*8, 2, stride=2)
        self.ConvBlock6 = block(dim*16, dim*8, strides=1)

        self.upv7 = nn.ConvTranspose2d(dim*8, dim*4, 2, stride=2)
        self.ConvBlock7 = block(dim*8, dim*4, strides=1)

        self.upv8 = nn.ConvTranspose2d(dim*4, dim*2, 2, stride=2)
        self.ConvBlock8 = block(dim*4, dim*2, strides=1)

        self.upv9 = nn.ConvTranspose2d(dim*2, dim, 2, stride=2)
        self.ConvBlock9 = block(dim*2, dim, strides=1)

        self.conv10 = nn.Conv2d(dim, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        conv1 = self.ConvBlock1(x)
        pool1 = self.pool1(conv1)

        conv2 = self.ConvBlock2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.ConvBlock3(pool2)
        pool3 = self.pool3(conv3)

        conv4 = self.ConvBlock4(pool3)
        pool4 = self.pool4(conv4)

        conv5 = self.ConvBlock5(pool4)

        up6 = self.upv6(conv5)
        up6 = f.pad(up6, (0, 1, 0, 1), mode='constant', value=0)
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.ConvBlock6(up6)

        up7 = self.upv7(conv6)
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.ConvBlock7(up7)

        up8 = self.upv8(conv7)
        up8 = f.pad(up8, (0, 1, 0, 1), mode='constant', value=0)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.ConvBlock8(up8)

        up9 = self.upv9(conv8)
        up9 = f.pad(up9, (0, 1, 0, 1), mode='constant', value=0)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.ConvBlock9(up9)

        out = self.conv10(conv9)
        #out = x[:,:2,:,:] + out

        return out