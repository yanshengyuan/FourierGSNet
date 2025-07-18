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

folder_path = "Val_Data"
outputPath = "test_set/"

png_files = glob.glob(os.path.join(folder_path, "*.png"))

for file in png_files:
    os.remove(file)
    print(f"Deleted: {file}")
    
npy_files = [os.path.basename(f) for f in glob.glob(os.path.join(folder_path, "*.npy"))]
npy_files.sort()

# Define paths and filenames for input
inputPathStr="./Input_Data/"
configFileStr="Config_AI_Data_Generator-train.dat"

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

nLight1000=Intensity(lightField,flag=2)
nLight1000 = nLight1000[start_idx:end_idx, start_idx:end_idx]
mpimg.imsave("lightsource.png",-nLight1000,cmap='Greys')
np.save("lightsource.npy", nLight1000)

# Prepare and apply CGH phase mask to field
cghFilename = config["cgh_data"]["cghFilename"]
cghBackgroundValue = config["cgh_data"].getint("cghBackgroundValue") 
cghGreyValues = config["cgh_data"].getint("cghGreyValues")
cghSize = config["cgh_data"].getfloat("cghSize")
cghPixelNumber = config["cgh_data"].getint("cghPixelNumber")

cghImageData = mpimg.imread(inputPathStr + cghFilename) 
cghPhaseData = 2*np.pi*(np.asarray(cghImageData[:,100:700])-cghBackgroundValue/cghGreyValues)

cghField=Begin(cghSize,wavelength,cghPixelNumber)
cghField=MultPhase(cghField,cghPhaseData)
cghField=Interpol(cghField, gridSize, gridPixelnumber, x_shift=0.0, y_shift=0.0, angle=0.0, magnif=1.0)
lightField=MultPhase(lightField,Phase(cghField))

# Prepare calculation of Zernike coefficients      
zernikeMaxOrder = config["zernike_coefficients"].getint("zernikeMaxOrder")
zernikeAmplitude = config["zernike_coefficients"].getfloat("zernikeAmplitude")
zernikeRadius = config["zernike_coefficients"].getfloat("zernikeRadius")
nollMin = config["zernike_coefficients"].getint("nollMin")


nollMax = np.sum(range(1,zernikeMaxOrder + 1))  # Maximum Noll index
nollRange=range(nollMin,nollMax+1)
zernikeCoeff=np.zeros(nollMax)

# Prepare main control loop
runMax = config["run_control"].getint("runMax")

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
causticPlanes.append(("-03-pst-", config["caustic_planes"].getfloat("prefocPlane") )) 

# Prepare intensity output
outputSize = config["data_output"].getfloat("outputSize")
outputPixelnumber = config["data_output"].getint("outputPixelnumber")

for runCount in range(len(npy_files)):
    print(runCount)
    runName = f"run{runCount:04}"
    outFileName_intensity = runName + "_intensity_"
    outFileName_phase = runName + "_phase"
    
    zernikeField = lightField
    zernikecoefficients = np.load(folder_path+"/"+npy_files[runCount])[3:]
    
    for countNoll in nollRange: 
        if countNoll>3:
            (nz,mz) = noll_to_zern(countNoll)
            zernikeCoeff[countNoll-1] = zernikecoefficients[countNoll-4]
            zernikeField = Zernike(zernikeField,nz,mz,zernikeRadius,zernikeCoeff[countNoll-1],units='rad')
        else:
            (nz,mz) = noll_to_zern(countNoll)
            zernikeCoeff[countNoll-1] = 0
            zernikeField = Zernike(zernikeField,nz,mz,zernikeRadius,zernikeCoeff[countNoll-1],units='rad')
    
    distField = zernikeField
    zernikeField = CircAperture(apertureRadius,0,0,zernikeField)
    phase = Phase(zernikeField)
    phase = phase[start_idx:end_idx, start_idx:end_idx]
    mpimg.imsave(outputPath+"/phase/img/" + outFileName_phase + ".png", phase, cmap='Greys')
    np.save(outputPath+"/phase/npy/" + outFileName_phase + ".npy", phase)
    # Prefocus field and apply aperture
    distField = Lens(f1,0,0,distField)
    distField = SmoothCircAperture(distField, apertureRadius, apertureSmoothWidth)
    
    for causticStr,causticPos in causticPlanes:
        cField=LensFresnel(distField,f2,focalLength+causticPos*zR)
        cField=Convert(cField)
        
        intensity=Intensity(cField,flag=2)
        intensity = intensity[start_idx:end_idx, start_idx:end_idx]
        mpimg.imsave(outputPath+"/intensity/img/" + outFileName_intensity + causticStr + ".png",-intensity,cmap='Greys')
        np.save(outputPath+"/intensity/npy/" + outFileName_intensity + ".npy", intensity)