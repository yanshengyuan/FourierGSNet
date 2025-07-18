import os

from dataset import DataLoaderTrain, DataLoaderVal, DataLoaderTrainCPU, DataLoaderValCPU
def get_training_data(rgb_dir):
    assert os.path.exists(rgb_dir)
    return DataLoaderTrain(rgb_dir)

def get_validation_data(rgb_dir):
    assert os.path.exists(rgb_dir)
    return DataLoaderVal(rgb_dir)

def get_training_dataCPU(rgb_dir):
    assert os.path.exists(rgb_dir)
    return DataLoaderTrain(rgb_dir)

def get_validation_dataCPU(rgb_dir):
    assert os.path.exists(rgb_dir)
    return DataLoaderVal(rgb_dir)