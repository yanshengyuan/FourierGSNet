"""
Shengyuan Yan, TU/e, Eindhoven, Netherlands, 17:00pm, 4/02/2025
"""

from M290_MachineSimu_GPU.optical_components_GPU.Thin_Lens import ThinLens
from M290_MachineSimu_GPU.optical_components_GPU.Apertures import SmoothEdgeAperture
from M290_MachineSimu_GPU.optical_components_GPU.Apertures import CircularAperture
from M290_MachineSimu_GPU.optical_components_GPU.Fresnel_BeamExpander import Expander_Fresnel
from M290_MachineSimu_GPU.complex_field_tools_GPU.complex_field_tools import *
from M290_MachineSimu_GPU.complex_field_tools_GPU.Simulation_grid import Spherer2Cartesian, Cartesian2Spherer

#from LightPipes import *

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import configparser
import numpy as np

import torch
from torch import nn

# path_lightsource = "../denserec30k/lightsource.npy"
class M290(nn.Module):
    def __init__(self, path_lightsource, device, imaging_plane='prefoc'):
        super(M290, self).__init__()

        inputPathStr="M290_MachineSimu_GPU/Input_Data/"
        configFileStr="Config_AI_Data_Generator.dat"

        config = configparser.ConfigParser()
        checkFile = config.read(inputPathStr+configFileStr)

        self.wavelength = config["field_initialization"].getfloat("wavelength")
        self.gridSize = config["field_aperture"].getfloat("apertureRadius")*2
        full_scene = config["field_initialization"].getfloat("gridSize")
        full_flatfield = config["field_initialization"].getfloat("gridPixelnumber")

        crop_ratio = self.gridSize/full_scene
        gridPixelnumber = int(full_flatfield*crop_ratio)
        beamDiameter = config["gaussian_beam"].getfloat("beamDiameter")
        beamWaist = beamDiameter/2

        beamMagnification = config["field_focussing"].getfloat("beamMagnification")
        self.focalLength = config["field_focussing"].getfloat("focalLength") / beamMagnification
        focalReduction = config["field_focussing"].getfloat("focalReduction")
                           
        self.f1=self.focalLength*focalReduction
        self.f2=self.f1*self.focalLength/(self.f1-self.focalLength)
        frac=self.focalLength/self.f1
        newSize=frac*self.gridSize

        self.apertureRadius = config["field_aperture"].getfloat("apertureRadius")
        self.apertureSmoothWidth = config["field_aperture"].getfloat("apertureSmoothWidth")

        focWaist = self.wavelength/np.pi*self.focalLength/beamWaist
        self.zR = np.pi*focWaist**2/self.wavelength

        self.causticPlanes = []
        self.causticPlanes.append(("pst", config["caustic_planes"].getfloat(imaging_plane+"Plane") ))
        
        self.initField = torch.zeros((gridPixelnumber, gridPixelnumber), dtype=torch.complex64).to(device)
        
        lightsource_npy = np.load(path_lightsource).astype(np.float32)
        self.lightsource = torch.from_numpy(lightsource_npy).to(device)
        
        self.nearField = torch.zeros((gridPixelnumber, gridPixelnumber), dtype=torch.complex64).to(device)
        self.nearField = SubIntensity(self.nearField, self.lightsource)
        
        self.Thin_lens = ThinLens(self.wavelength)
        self.Smooth_aperture = SmoothEdgeAperture(self.apertureRadius, self.apertureSmoothWidth)
        self.BeamExpand_propagator = Expander_Fresnel(self.wavelength)
        self.Spherer2Cartesian = Spherer2Cartesian(self.wavelength)
        self.Cartesian2Spherer = Cartesian2Spherer(self.wavelength)
        
    def M290_forward(self, field):
        
        field = self.Thin_lens(field, self.f1, self.gridSize)
        field = self.Smooth_aperture(field, self.gridSize)
        field, new_size, curvature = self.BeamExpand_propagator(field, self.f2, 
                                           self.focalLength+self.causticPlanes[0][1]*self.zR, self.gridSize, 0)
        field, final_size, final_curvature = self.Spherer2Cartesian(field, new_size, curvature)
        
        return field, final_size, final_curvature
    
    def M290_inverse(self, field, size, curvature):
        
        field, final_size, final_curvature = self.Cartesian2Spherer(field, size, curvature)
        field, new_size, new_curvature = self.BeamExpand_propagator(field, self.f2, 
                                           -(self.focalLength+self.causticPlanes[0][1]*self.zR), final_size, final_curvature)
        field = self.Smooth_aperture(field, new_size)
        field = self.Thin_lens(field, -self.f1, new_size)
        field = CircularAperture(field, self.apertureRadius, new_size)
        
        return field
    
    def forward(self, field, curvature, simulation, size=None):
        
        if simulation not in ['M290 machine Forward simulation', 'M290 machine Inverse simulation']:
            print("Wrong simulation!")
            
            return None
        
        if(curvature==0 and size==None and simulation=='M290 machine Forward simulation'):
            field, size, curvature = self.M290_forward(field)
            
            return field, size, curvature
            
        elif(curvature!=0 and size!=None and simulation=='M290 machine Inverse simulation'):
            field = self.M290_inverse(field, size, curvature)
            
            return field