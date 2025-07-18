import numpy as np
import os
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import random

##################################################################################################
class DataLoaderTrain(Dataset):
    def __init__(self, rgb_dir):
        super(DataLoaderTrain, self).__init__()
        
        gt_dir = os.path.join('train', 'phase', 'npy')
        input_dir = os.path.join('train', 'intensity', 'npy')
        lightsource_dir = os.path.join('train', 'lightsource', 'npy')
        
        I_files = sorted(os.listdir(os.path.join(rgb_dir, input_dir)))
        Phi_files = sorted(os.listdir(os.path.join(rgb_dir, gt_dir)))
        lightsource_files = sorted(os.listdir(os.path.join(rgb_dir, lightsource_dir)))
        
        self.I_filenames = [os.path.join(rgb_dir, input_dir, x) for x in I_files]
        self.Phi_filenames = [os.path.join(rgb_dir, gt_dir, x) for x in Phi_files]
        self.lightsource_filenames = [os.path.join(rgb_dir, lightsource_dir, x) for x in lightsource_files]
        self.tar_size = len(self.I_filenames)

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        I=np.load(self.I_filenames[index]).astype(np.float32)[317:445, 317:445]
        Phi=np.load(self.Phi_filenames[index]).astype(np.float32)
        lightsource=np.load(self.lightsource_filenames[index]).astype(np.float32)
        I=torch.tensor(I)
        Phi=torch.tensor(Phi)
        lightsource=torch.tensor(lightsource)
        I=I.unsqueeze(0)
        Phi=Phi.unsqueeze(0)
        #print(self.I_filenames[index])
        #print(self.Phi_filenames[index])

        return I, Phi

##################################################################################################
class DataLoaderTrainCPU(Dataset):
    def __init__(self, rgb_dir):
        super(DataLoaderTrain, self).__init__()
        
        gt_dir = os.path.join('train', 'phase', 'npy')
        input_dir = os.path.join('train', 'intensity', 'npy')
        
        I_files = sorted(os.listdir(os.path.join(rgb_dir, input_dir)))
        Phi_files = sorted(os.listdir(os.path.join(rgb_dir, gt_dir)))
        
        self.I_filenames = [os.path.join(rgb_dir, input_dir, x) for x in I_files]
        self.Phi_filenames = [os.path.join(rgb_dir, gt_dir, x) for x in Phi_files]
        self.tar_size = len(self.I_filenames)

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        I=np.load(self.I_filenames[index]).astype(np.float32)
        Phi=np.load(self.Phi_filenames[index]).astype(np.float32)

        return I, Phi

##################################################################################################
class DataLoaderVal(Dataset):
    def __init__(self, rgb_dir):
        super(DataLoaderVal, self).__init__()
        
        gt_dir = os.path.join('test', 'phase', 'npy')
        input_dir = os.path.join('test', 'intensity', 'npy')
        lightsource_dir = os.path.join('train', 'lightsource', 'npy')
        
        I_files = sorted(os.listdir(os.path.join(rgb_dir, input_dir)))
        Phi_files = sorted(os.listdir(os.path.join(rgb_dir, gt_dir)))
        lightsource_files = sorted(os.listdir(os.path.join(rgb_dir, lightsource_dir)))
        
        self.I_filenames = [os.path.join(rgb_dir, input_dir, x) for x in I_files]
        self.Phi_filenames = [os.path.join(rgb_dir, gt_dir, x) for x in Phi_files]
        self.lightsource_filenames = [os.path.join(rgb_dir, lightsource_dir, x) for x in lightsource_files]
        self.tar_size = len(self.I_filenames)

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        I=np.load(self.I_filenames[index]).astype(np.float32)[317:445, 317:445]
        Phi=np.load(self.Phi_filenames[index]).astype(np.float32)
        lightsource=np.load(self.lightsource_filenames[index]).astype(np.float32)
        I=torch.tensor(I)
        Phi=torch.tensor(Phi)
        lightsource=torch.tensor(lightsource)
        I=I.unsqueeze(0)
        Phi=Phi.unsqueeze(0)
        #print(self.I_filenames[index])
        #print(self.Phi_filenames[index])

        return I, Phi
    
##################################################################################################
class DataLoaderValCPU(Dataset):
    def __init__(self, rgb_dir):
        super(DataLoaderValCPU, self).__init__()
        
        gt_dir = os.path.join('test', 'phase', 'npy')
        input_dir = os.path.join('test', 'intensity', 'npy')
        
        I_files = sorted(os.listdir(os.path.join(rgb_dir, input_dir)))
        self.Phi_files = sorted(os.listdir(os.path.join(rgb_dir, gt_dir)))
        
        self.I_filenames = [os.path.join(rgb_dir, input_dir, x) for x in I_files]
        self.Phi_filenames = [os.path.join(rgb_dir, gt_dir, x) for x in self.Phi_files]
        self.tar_size = len(self.I_filenames)

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        I=np.load(self.I_filenames[index]).astype(np.float32)
        Phi=np.load(self.Phi_filenames[index]).astype(np.float32)
        name=self.Phi_files[index]

        return I, Phi