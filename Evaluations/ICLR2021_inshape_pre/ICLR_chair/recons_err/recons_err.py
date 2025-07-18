import os
import glob
from LightPipes import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import math
import random
import configparser
import numpy as np
from UserFunctions.UserFunctions import SmoothStep
from UserFunctions.UserFunctions import SmoothCircAperture
import pandas as pd
from PIL import Image
import seaborn as sns

# Define paths and filenames for input
inputPathStr="./Input_Data/"
configFileStr="Config_AI_Data_Generator.dat"

# Open data generator config file
config = configparser.ConfigParser()
checkFile = config.read(inputPathStr+configFileStr)

# Define initial field
wavelength = config["field_initialization"].getfloat("wavelength")
gridSize = config["field_initialization"].getfloat("gridSize")
gridPixelnumber = config["field_initialization"].getint("gridPixelnumber")
beamDiameter = config["gaussian_beam"].getfloat("beamDiameter")
beamWaist = beamDiameter/2
#Prepare field aperture
apertureRadius = config["field_aperture"].getfloat("apertureRadius")
apertureSmoothWidth = config["field_aperture"].getfloat("apertureSmoothWidth")

lightField = Begin(gridSize,wavelength,gridPixelnumber)
lightField = GaussBeam(lightField, beamWaist, n = 0, m = 0, x_shift = 0, y_shift=0, tx=0, ty=0, doughnut=False, LG=True)

crop_ratio = (apertureRadius*2)/gridSize
crop_size = int(gridPixelnumber*crop_ratio)
print(crop_ratio)
print(gridPixelnumber)
print(crop_size)
start_idx = (gridPixelnumber - crop_size) // 2
end_idx = start_idx + crop_size

#Prepare field focusing 
beamMagnification = config["field_focussing"].getfloat("beamMagnification")
focalLength = config["field_focussing"].getfloat("focalLength") / beamMagnification
focalReduction = config["field_focussing"].getfloat("focalReduction")
                   
f1=focalLength*focalReduction
f2=f1*focalLength/(f1-focalLength)
frac=focalLength/f1
newSize=frac*gridSize
newExtent=[-newSize/2/mm,newSize/2/mm,-newSize/2/mm,newSize/2/mm]

# Prepare propagation of field to caustic planes
focWaist = wavelength/np.pi*focalLength/beamWaist  # Focal Gaussian beam waist
zR = np.pi*focWaist**2/wavelength # Rayleigh range focused Gaussian beam

causticPlanes = []
causticPlanes.append(("-01-pre-", config["caustic_planes"].getfloat("prefocPlane") )) 

# Prepare intensity output
outputSize = config["data_output"].getfloat("outputSize")
outputPixelnumber = config["data_output"].getint("outputPixelnumber")




gt_folder = "../I_gt/npy/"
gt_files = [os.path.basename(f) for f in glob.glob(os.path.join(gt_folder, "*.npy"))]
gt_files.sort()

pred_folder = "../Phi_pred/npy/"
pred_files = [os.path.basename(f) for f in glob.glob(os.path.join(pred_folder, "*.npy"))]
pred_files.sort()

distField = lightField
pad_size = (gridPixelnumber - crop_size) // 2

mae_list = []
for i in range(len(pred_files)):
    
    pred_phase = np.load(pred_folder+pred_files[i])
    phase = Phase(distField)
    phase[pad_size:pad_size + crop_size, pad_size:pad_size + crop_size] = pred_phase
    distField = SubPhase(phase, distField)
    distField = Lens(f1,0,0,distField)
    distField = SmoothCircAperture(distField, apertureRadius, apertureSmoothWidth)
    cField=LensFresnel(distField,f2,focalLength+causticPlanes[0][1]*zR)
    cField=Convert(cField)
    
    pred_I = Intensity(cField,flag=2)
    pred_I = pred_I[start_idx:end_idx, start_idx:end_idx]
    mpimg.imsave("pred/" + pred_files[i][:-4] + ".png",-pred_I,cmap='Greys')
    
    gt_I = np.load(gt_folder+gt_files[i])
    mae = np.mean(np.abs(pred_I - gt_I))
    print(mae)
    mae_list.append(mae)
    
    print(i)
    #if(i==10): break
    
mean_mae = np.mean(mae_list)
print(f"Mean MAE: {mean_mae}")

sns.kdeplot(mae_list, fill=True, bw_adjust=0.5)

# Labels and title
plt.xlabel("Mean Absolute Error (MAE)")
plt.ylabel("Density")
plt.title("KDE Plot of MAE metric across 100 test samples")
plt.savefig("KDE_MAE.png", dpi=300)
plt.close()

mae_list=np.array(mae_list)
np.save("../recons.npy", mae_list)