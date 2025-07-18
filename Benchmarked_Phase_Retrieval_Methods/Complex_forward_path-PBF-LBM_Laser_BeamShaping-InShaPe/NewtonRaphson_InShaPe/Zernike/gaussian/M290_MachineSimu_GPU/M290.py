"""
Shengyuan Yan, TU/e, Eindhoven, Netherlands, 01:57am, 4/08/2025
"""

from M290_MachineSimu_GPU.optical_components_GPU.Thin_Lens import ThinLens
from M290_MachineSimu_GPU.optical_components_GPU.Apertures import SmoothEdgeAperture
from M290_MachineSimu_GPU.optical_components_GPU.Apertures import CircularAperture
from M290_MachineSimu_GPU.optical_components_GPU.Fresnel_BeamExpander import Expander_Fresnel
from M290_MachineSimu_GPU.complex_field_tools_GPU.complex_field_tools import *
from M290_MachineSimu_GPU.complex_field_tools_GPU.Simulation_grid import MeshGrid_Polar, Spherer2Cartesian, Cartesian2Spherer
from M290_MachineSimu_GPU.complex_field_tools_GPU.Zernike_polynomials import Zernike_Polynomial, noll_to_zern

#from LightPipes import *

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import configparser
from pathlib import Path
import numpy as np

import torch
from torch import nn

# path_lightsource = "../denserec30k/lightsource.npy"
class M290(nn.Module):
    def __init__(self, beamshape, path_lightsource, device, imaging_plane='prefoc'):
        super(M290, self).__init__()

        inputPathStr="M290_MachineSimu_GPU/Configs/Input_Data_"+beamshape+"/"
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
        
        cghFilename = config["cgh_data"]["cghFilename"]
        cghBackgroundValue = config["cgh_data"].getint("cghBackgroundValue") 
        cghGreyValues = config["cgh_data"].getint("cghGreyValues")
        self.cghSize = config["cgh_data"].getfloat("cghSize")
        self.cghPixelNumber = config["cgh_data"].getint("cghPixelNumber")

        cghImageData = mpimg.imread(inputPathStr + cghFilename)
        if(beamshape=='Chair' or beamshape=='gaussian'):
            cghPhaseData = 2*np.pi*(np.asarray(cghImageData[:,:,0])-cghBackgroundValue/cghGreyValues)
        else:
            cghPhaseData = 2*np.pi*(np.asarray(cghImageData[:,100:700])-cghBackgroundValue/cghGreyValues)

        cghField = np.ones((self.cghPixelNumber, self.cghPixelNumber), dtype=np.complex64)
        if(beamshape != 'gaussian'):
            cghField = SubPhase(cghField,cghPhaseData)

        cghField = field_interpolate(cghField, self.cghSize, self.gridSize, self.cghPixelNumber, gridPixelnumber,
                                          x_shift=0.0, y_shift=0.0, angle=0.0, magnif=1.0)
        SLM_Phase_Mask = Phase(cghField)
        SLM_Phase_Mask = torch.from_numpy(SLM_Phase_Mask).to(device)
        self.SLM_Phase_Mask = SLM_Phase_Mask
        
        self.zernikeAmplitude = config["zernike_coefficients"].getfloat("zernikeAmplitude")
        self.zernikeRadius = config["zernike_coefficients"].getfloat("zernikeRadius")
        path = Path(path_lightsource)
        directory = str(path.parent) + "/"
        exmp_ZernikeCoeff_path = directory + "Zernike_coefficients/" + "run0000_zernikeCoeff.npy"
        self.ZernikeCoeff_init = np.load(exmp_ZernikeCoeff_path)
        self.initField = np.zeros((gridPixelnumber, gridPixelnumber), dtype=np.complex64)
        polar_radius, polar_angle = MeshGrid_Polar(self.initField, self.gridSize)
        
        nollMax = len(self.ZernikeCoeff_init)
        nollMin = 4
        noll_range = range(nollMin,nollMax+1)
        
        self.Zernike_mode_list = []
        for Noll_ind in noll_range:
            (nz,mz) = noll_to_zern(Noll_ind)
            Zernike_mode = Zernike_Polynomial(self.wavelength, self.gridSize, polar_radius, polar_angle, 
                                              nz, mz, self.zernikeRadius, units='rad')
            Zernike_mode = Zernike_mode.astype(np.complex64)
            Zernike_mode = torch.from_numpy(Zernike_mode).to(device)
            self.Zernike_mode_list.append(Zernike_mode)
        self.Zernike_modes = torch.stack(self.Zernike_mode_list).unsqueeze(0)

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
        
        lightsource_npy = np.load(path_lightsource).astype(np.float32)
        self.lightsource = torch.from_numpy(lightsource_npy).to(device)
        
        self.nearField = torch.zeros((gridPixelnumber, gridPixelnumber), dtype=torch.complex64).to(device)
        self.nearField = SubIntensity(self.nearField, self.lightsource)
        self.nearField = SubPhase(self.nearField, self.SLM_Phase_Mask)
        
        self.Thin_lens = ThinLens(self.wavelength)
        self.Smooth_aperture = SmoothEdgeAperture(self.apertureRadius, self.apertureSmoothWidth)
        self.BeamExpand_propagator = Expander_Fresnel(self.wavelength)
        self.Spherer2Cartesian = Spherer2Cartesian(self.wavelength)
        self.Cartesian2Spherer = Cartesian2Spherer(self.wavelength)
        
        self.zernike_coeffs = nn.Parameter(torch.zeros(12, dtype=torch.float32))
        
    def M290_forward(self, field):
        
        zernike_coefficients = self.zernike_coeffs.unsqueeze(-1).unsqueeze(-1)
        weighted_Zernike_modes = zernike_coefficients * self.Zernike_modes
        weighted_angular_spectrum = torch.exp(1j*weighted_Zernike_modes)
        aberration = weighted_angular_spectrum.prod(dim=1)
        field = field * aberration
        
        aberrated_pupil_field = CircularAperture(field, self.apertureRadius, self.gridSize)
        aberrated_SLM_phase = Phase(aberrated_pupil_field)
        
        field = self.Thin_lens(field, self.f1, self.gridSize)
        field = self.Smooth_aperture(field, self.gridSize)
        field, new_size, curvature = self.BeamExpand_propagator(field, self.f2, 
                                           self.focalLength+self.causticPlanes[0][1]*self.zR, self.gridSize, 0)
        field, final_size, final_curvature = self.Spherer2Cartesian(field, new_size, curvature)
        
        return field, final_size, final_curvature, aberrated_SLM_phase
    
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
            field, size, curvature, phase = self.M290_forward(field)
            
            return field, size, curvature, phase
            
        elif(curvature!=0 and size!=None and simulation=='M290 machine Inverse simulation'):
            field = self.M290_inverse(field, size, curvature)
            
            return field