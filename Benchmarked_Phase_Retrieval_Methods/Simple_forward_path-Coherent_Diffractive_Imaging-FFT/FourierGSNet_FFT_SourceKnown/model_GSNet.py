import torch
import torch.nn as nn
import torch.nn.functional as f
from einops import rearrange
import numpy as np
import matplotlib.pyplot as plt
import os

class FFT_layer(nn.Module):
    def __init__(self, args):
        super(FFT_layer, self).__init__()
        self.phase_updating_network = Condition_UNet()
        
    def forward(self, source_phase, pupil_field, Fourier_intensity, pupil_intensity, epoch=None, num_batch=None, num_layer=None, val=False, output_dir=None):

        bs_1 = False
        if(Fourier_intensity.dim()==2):
            bs_1 = True
        if(bs_1==True):
            Fourier_intensity = Fourier_intensity.unsqueeze(0)
        
        crop = Fourier_intensity[: ,317:445, 317:445]
        
        Fourier_intensity = Fourier_intensity.unsqueeze(1)
        Fourier_intensity = f.interpolate(Fourier_intensity, size=(128, 128), mode='bilinear', align_corners=False)
        Fourier_intensity = Fourier_intensity.squeeze()
        #print(Fourier_intensity.shape)
        #plt.imsave("resized.png", Fourier_intensity[0].detach().cpu().numpy(), cmap='gray')
        #plt.imsave("cropped.png", crop[0].detach().cpu().numpy(), cmap='gray')
        
        Fourier_amp = torch.sqrt(Fourier_intensity)
        pupil_amp = torch.sqrt(pupil_intensity)
        Fourier_field = torch.fft.fft2(pupil_field)
        Fourier_field = torch.fft.fftshift(Fourier_field)
        Fourier_real = Fourier_field.real
        Fourier_imag = Fourier_field.imag
        Fourier_phase = torch.angle(Fourier_field)
        Fourier_field = Fourier_amp*torch.exp(Fourier_phase * 1j)
        
        pupil_field = torch.fft.ifftshift(Fourier_field)
        pupil_field = torch.fft.ifft2(pupil_field)
        pupil_phase = torch.angle(pupil_field)
        pupil_field = pupil_amp*torch.exp(pupil_phase * 1j)
        
        Fourier_phase = Fourier_phase.unsqueeze(1)
        pupil_phase = pupil_phase.unsqueeze(1)
        crop = crop.unsqueeze(1)
        Fourier_real = Fourier_real.unsqueeze(1)
        Fourier_imag = Fourier_imag.unsqueeze(1)
        
        Input = torch.stack((source_phase, crop), dim=1)
        condition = torch.stack((pupil_phase, Fourier_real, Fourier_imag), dim=1)
        attention = torch.stack((source_phase, pupil_phase, Fourier_real, Fourier_imag), dim=1)
        if(bs_1==False):
            condition = condition.squeeze()
            Input = Input.squeeze()
            attention = attention.squeeze()
        elif(bs_1==True):
            condition = condition.squeeze(2)
            Input = Input.squeeze(2)
            attention = attention.squeeze(2)
        
        source_phase = self.phase_updating_network(Input, condition, attention, epoch, num_batch, num_layer, val, output_dir)
        
        return pupil_field, source_phase
    
class GSNet_Fourier(nn.Module):
    def __init__(self, gen_phase_init, gs_phase_init, args, num_layers=10):
        super(GSNet_Fourier, self).__init__()
        self.layers = nn.ModuleList()
        self.input_layer = FFT_layer(args)
        self.gen_phase_init = gen_phase_init
        self.gs_phase_init = gs_phase_init
        
        for _ in range(num_layers):
            layer = FFT_layer(args)
            self.layers.append(layer)
            
        #print(self.gen_phase_init)
        #print(self.gs_phase_init)
        
    def forward(self, source_intensity, image_intensity, epoch=None, num_batch=None, val=False, output_dir=None):
        if(val==True):
            attention_folder = os.path.join(output_dir, str(epoch), "attention_map")
            feature_folder = os.path.join(output_dir, str(epoch), "feature_map")
            trunk_folder = os.path.join(output_dir, str(epoch), "trunk_map")
            os.makedirs(attention_folder, exist_ok=True)
            os.makedirs(feature_folder, exist_ok=True)
            os.makedirs(trunk_folder, exist_ok=True)
        
        source_amp = torch.sqrt(source_intensity)
        init_field = source_amp*torch.exp(self.gs_phase_init * 1j)
        gen_phase_init = self.gen_phase_init.unsqueeze(1)
        F, phase = self.input_layer(gen_phase_init, init_field, image_intensity, source_intensity, epoch, num_batch, 0, val, output_dir)
        cnt = 1
        for layer in self.layers:
            F, phase = layer(phase, F, image_intensity, source_intensity, epoch, num_batch, cnt, val, output_dir)
            cnt += 1
        
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
    
class Condition_Encoder(nn.Module):
    def __init__(self, block=ConvBlock,dim=32):
        super(Condition_Encoder, self).__init__()

        self.dim = dim
        self.ConvBlock1 = ConvBlock(3, dim, strides=1)
        self.pool1 = nn.Conv2d(dim,dim,kernel_size=4, stride=2, padding=1)

        self.ConvBlock2 = block(dim, dim*2, strides=1)
        self.pool2 = nn.Conv2d(dim*2,dim*2,kernel_size=4, stride=2, padding=1)
       
        self.ConvBlock3 = block(dim*2, dim*4, strides=1)
        self.pool3 = nn.Conv2d(dim*4,dim*4,kernel_size=4, stride=2, padding=1)
       
        self.ConvBlock4 = block(dim*4, dim*8, strides=1)
        self.pool4 = nn.Conv2d(dim*8, dim*8,kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        conv1 = self.ConvBlock1(x)
        pool1 = self.pool1(conv1)

        conv2 = self.ConvBlock2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.ConvBlock3(pool2)
        pool3 = self.pool3(conv3)

        conv4 = self.ConvBlock4(pool3)
        condition_feature = self.pool4(conv4)

        return condition_feature
    
class Attention_Encoder(nn.Module):
    def __init__(self, block=ConvBlock,dim=32):
        super(Attention_Encoder, self).__init__()

        self.dim = dim
        self.ConvBlock1 = ConvBlock(4, dim, strides=1)
        self.pool1 = nn.Conv2d(dim,dim,kernel_size=4, stride=2, padding=1)

        self.ConvBlock2 = block(dim, dim*2, strides=1)
        self.pool2 = nn.Conv2d(dim*2,dim*2,kernel_size=4, stride=2, padding=1)
       
        self.ConvBlock3 = block(dim*2, dim*4, strides=1)
        self.pool3 = nn.Conv2d(dim*4,dim*4,kernel_size=4, stride=2, padding=1)
       
        self.ConvBlock4 = block(dim*4, dim*8, strides=1)
        self.pool4 = nn.Conv2d(dim*8, dim*24,kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        conv1 = self.ConvBlock1(x)
        pool1 = self.pool1(conv1)

        conv2 = self.ConvBlock2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.ConvBlock3(pool2)
        pool3 = self.pool3(conv3)

        conv4 = self.ConvBlock4(pool3)
        condition_feature = self.pool4(conv4)

        return condition_feature


class Condition_UNet(nn.Module):
    def __init__(self, block=ConvBlock,dim=32):
        super(Condition_UNet, self).__init__()
        
        self.condition_feature_encoder = Condition_Encoder()
        self.attention_map_encoder = Attention_Encoder()

        self.dim = dim
        self.ConvBlock1 = ConvBlock(2, dim, strides=1)
        self.pool1 = nn.Conv2d(dim,dim,kernel_size=4, stride=2, padding=1)

        self.ConvBlock2 = block(dim, dim*2, strides=1)
        self.pool2 = nn.Conv2d(dim*2,dim*2,kernel_size=4, stride=2, padding=1)
       
        self.ConvBlock3 = block(dim*2, dim*4, strides=1)
        self.pool3 = nn.Conv2d(dim*4,dim*4,kernel_size=4, stride=2, padding=1)
       
        self.ConvBlock4 = block(dim*4, dim*8, strides=1)
        self.pool4 = nn.Conv2d(dim*8, dim*8,kernel_size=4, stride=2, padding=1)

        self.ConvBlock5 = block(dim*16, dim*16, strides=1)
        #self.ConvBlock5 = block(dim*8, dim*16, strides=1)

        self.upv6 = nn.ConvTranspose2d(dim*16, dim*8, 2, stride=2)
        self.ConvBlock6 = block(dim*16, dim*8, strides=1)

        self.upv7 = nn.ConvTranspose2d(dim*8, dim*4, 2, stride=2)
        self.ConvBlock7 = block(dim*8, dim*4, strides=1)

        self.upv8 = nn.ConvTranspose2d(dim*4, dim*2, 2, stride=2)
        self.ConvBlock8 = block(dim*4, dim*2, strides=1)

        self.upv9 = nn.ConvTranspose2d(dim*2, dim, 2, stride=2)
        self.ConvBlock9 = block(dim*2, dim, strides=1)

        self.conv10 = nn.Conv2d(dim, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x, condition, attention, epoch=None, num_batch=None, num_layer=None, val=False, output_dir=None):
        
        attention_map = self.attention_map_encoder(attention)
        attention_map = rearrange(attention_map, 'b (c1 c2) h w -> b c1 (c2 h w)', c1=3)
        attention_map = rearrange(attention_map, 'b c (h w) -> b c h w', h=128, w=128)
        condition = condition*attention_map
        
        condition_feature = self.condition_feature_encoder(condition)
        
        if(val==True):
            attention_folder = os.path.join(output_dir, str(epoch), "attention_map")
            feature_folder = os.path.join(output_dir, str(epoch), "feature_map")
            trunk_folder = os.path.join(output_dir, str(epoch), "trunk_map")
            attention_map = rearrange(attention_map, 'b (1 groups) h w -> b 1 h (groups w)', groups=3)
            for i in range(attention_map.shape[0]):
                heat_map=attention_map[i].squeeze().cpu().numpy()
                plt.imshow(heat_map, cmap='hot')
                plt.colorbar()
                attention_img_name = "layer"+str(num_layer)+"_batch"+str(num_batch)+"_"+str(i)+".png"
                path = os.path.join(attention_folder, attention_img_name)
                plt.savefig(path)
                plt.close()
        
        conv1 = self.ConvBlock1(x)
        pool1 = self.pool1(conv1)

        conv2 = self.ConvBlock2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.ConvBlock3(pool2)
        pool3 = self.pool3(conv3)

        conv4 = self.ConvBlock4(pool3)
        pool4 = self.pool4(conv4)
        
        if(val==True):
            trunk_feature = rearrange(pool4, 'b c h w -> b 1 c (h w)')
            for i in range(trunk_feature.shape[0]):
                heat_map=trunk_feature[i].squeeze().cpu().numpy()
                plt.imshow(heat_map, cmap='hot')
                plt.colorbar()
                feature_img_name = "layer"+str(num_layer)+"_batch"+str(num_batch)+"_"+str(i)+".png"
                path = os.path.join(trunk_folder, feature_img_name)
                plt.savefig(path)
                plt.close()
        
        pool4 = torch.cat([pool4, condition_feature], 1)
        #pool4 = pool4 + condition_feature
        
        if(val==True):
            condition_feature = rearrange(condition_feature, 'b c h w -> b 1 c (h w)')
            for i in range(condition_feature.shape[0]):
                heat_map=condition_feature[i].squeeze().cpu().numpy()
                plt.imshow(heat_map, cmap='hot')
                plt.colorbar()
                feature_img_name = "layer"+str(num_layer)+"_batch"+str(num_batch)+"_"+str(i)+".png"
                path = os.path.join(feature_folder, feature_img_name)
                plt.savefig(path)
                plt.close()

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