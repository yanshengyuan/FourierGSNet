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
        
        gt_dir = 'GTs'
        input_dir = 'Zernike_coefficients'
        
        intensity_files = sorted(os.listdir(os.path.join(rgb_dir, gt_dir)))
        zernike_files = sorted(os.listdir(os.path.join(rgb_dir, input_dir)))
        zernike_file_list = []
        intensity_file_list = []
        
        for file in zernike_files:
            zernike_file_list.append(file)
        for file in intensity_files:
            intensity_file_list.append(file)
        
        self.filenames = intensity_files
        self.zernike_filenames = [os.path.join(rgb_dir, input_dir, x) for x in zernike_file_list]
        self.intensity_filenames = [os.path.join(rgb_dir, gt_dir, x) for x in intensity_file_list]

        self.tar_size = len(intensity_files)

    def __len__(self):
        
        return self.tar_size

    def __getitem__(self, index):
        
        zernike = torch.from_numpy(np.float32(np.load(self.zernike_filenames[index])[3:]))
        intensity = torch.from_numpy(np.float32(np.load(self.intensity_filenames[index])))
        name = self.filenames[index]

        return intensity, zernike, name