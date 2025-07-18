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
        
        gt_dir = 'Zernike_coefficients'
        input_dir = 'Zernike_coefficients'
        
        files = sorted(os.listdir(os.path.join(rgb_dir, gt_dir)))
        zernike_files = []
        for file in files:
            zernike_files.append(file)
        
        self.filenames = zernike_files
        self.zernike_filenames = [os.path.join(rgb_dir, gt_dir, x) for x in zernike_files]

        self.tar_size = len(zernike_files)

    def __len__(self):
        
        return self.tar_size

    def __getitem__(self, index):
        
        zernike = torch.from_numpy(np.float32(np.load(self.zernike_filenames[index])[3:]))
        name = self.filenames[index]

        return zernike, name