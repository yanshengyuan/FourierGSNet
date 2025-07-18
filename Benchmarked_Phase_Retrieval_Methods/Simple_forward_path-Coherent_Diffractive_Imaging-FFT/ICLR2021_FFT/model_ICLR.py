import torch
import torch.nn as nn
import torch.nn.functional as f
from einops import rearrange
import numpy as np
import matplotlib.pyplot as plt

class FFT_layer(nn.Module):
    def __init__(self, args):
        super(FFT_layer, self).__init__()
        
        self.phase_updating_network = UNet_original()
        
    def forward(self, phase, intens_far, intens_near, defocus_phase_mask):
        
        angular_mask = torch.exp(defocus_phase_mask * 1j)
        
        bs_1 = False
        if(intens_far.dim()==2):
            bs_1 = True
        if(bs_1==True):
            intens_far = intens_far.unsqueeze(0)
            intens_near = intens_near.unsqueeze(0)
        
        crop = intens_far[: ,317:445, 317:445]
        
        intens_far = intens_far.unsqueeze(1)
        intens_far = f.interpolate(intens_far, size=(128, 128), mode='bilinear', align_corners=False)
        intens_far = intens_far.squeeze()
        
        near_amp = torch.sqrt(intens_near)
        near_field = near_amp*torch.exp(phase * 1j)
        near_field = near_field * angular_mask
        
        far_field = torch.fft.fft2(near_field)
        far_field = torch.fft.fftshift(far_field)
        
        real_far = far_field.real
        real_far = real_far.unsqueeze(1)
        imag_far = far_field.imag
        imag_far = imag_far.unsqueeze(1)
        
        far_amp = torch.sqrt(intens_far)
        phase_far = torch.angle(far_field)
        far_field = far_amp*torch.exp(phase_far * 1j)
        
        near_field = torch.fft.ifftshift(far_field)
        near_field = torch.fft.ifft2(near_field)
        near_field /= angular_mask
        
        phase=torch.angle(near_field)
        phase = phase.unsqueeze(1)
        intens_far = intens_far.unsqueeze(1)
        crop = crop.unsqueeze(1)
        tensor = torch.stack((real_far, imag_far, crop, phase), dim=1)
        if(bs_1==False):
            tensor = tensor.squeeze()
        elif(bs_1==True):
            tensor = tensor.squeeze(2)
        
        phase = self.phase_updating_network(tensor)
        phase = phase.squeeze()
        
        return phase
    
class GSNet_FFT(nn.Module):
    def __init__(self, phase_init, defocus_phase_mask, args, num_layers=10):
        super(GSNet_FFT, self).__init__()
        self.layers = nn.ModuleList()
        self.input_layer = FFT_layer(args)
        self.phase_init = phase_init
        self.defocus_kernel = defocus_phase_mask
        
        for _ in range(num_layers):
            layer = FFT_layer(args)
            self.layers.append(layer)
        
    def forward(self, intens_near, intens_far):
        
        phase = self.input_layer(self.phase_init, intens_far, intens_near, self.defocus_kernel)
        for layer in self.layers:
            phase = layer(phase, intens_far, intens_near, self.defocus_kernel)
        
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
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.ConvBlock6(up6)

        up7 = self.upv7(conv6)
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.ConvBlock7(up7)

        up8 = self.upv8(conv7)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.ConvBlock8(up8)

        up9 = self.upv9(conv8)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.ConvBlock9(up9)

        out = self.conv10(conv9)

        return out