"""
Shengyuan Yan, TU/e, Eindhoven, Netherlands, 19:00pm, 3/19/2025
"""

# Import required modules
from LightPipes import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import configparser
import numpy as np
from M290_machine_simu.UserFunctions.UserFunctions import SmoothStep
from M290_machine_simu.UserFunctions.UserFunctions import SmoothCircAperture

# path_lightsource = "../denserec30k/lightsource.npy"
class M290:
    def __init__(self, path_lightsource):
        inputPathStr="M290_machine_simu/Input_Data/"
        configFileStr="Config_AI_Data_Generator.dat"

        config = configparser.ConfigParser()
        checkFile = config.read(inputPathStr+configFileStr)

        wavelength = config["field_initialization"].getfloat("wavelength")
        gridSize = config["field_aperture"].getfloat("apertureRadius")*2
        full_scene = config["field_initialization"].getfloat("gridSize")
        full_flatfield = config["field_initialization"].getfloat("gridPixelnumber")

        crop_ratio = gridSize/full_scene
        gridPixelnumber = int(full_flatfield*crop_ratio)
        beamDiameter = config["gaussian_beam"].getfloat("beamDiameter")
        beamWaist = beamDiameter/2

        beamMagnification = config["field_focussing"].getfloat("beamMagnification")
        self.focalLength = config["field_focussing"].getfloat("focalLength") / beamMagnification
        focalReduction = config["field_focussing"].getfloat("focalReduction")
                           
        self.f1=self.focalLength*focalReduction
        self.f2=self.f1*self.focalLength/(self.f1-self.focalLength)
        frac=self.focalLength/self.f1
        newSize=frac*gridSize
        newExtent=[-newSize/2/mm,newSize/2/mm,-newSize/2/mm,newSize/2/mm]

        self.apertureRadius = config["field_aperture"].getfloat("apertureRadius")
        self.apertureSmoothWidth = config["field_aperture"].getfloat("apertureSmoothWidth")

        focWaist = wavelength/np.pi*self.focalLength/beamWaist
        self.zR = np.pi*focWaist**2/wavelength

        self.causticPlanes = []
        self.causticPlanes.append(("pst", config["caustic_planes"].getfloat("postfocPlane") ))

        self.initField = Begin(gridSize,wavelength,gridPixelnumber)
        
        self.lightsource = np.load(path_lightsource)
        self.nearField = Begin(gridSize,wavelength,gridPixelnumber)
        self.nearField = SubIntensity(self.nearField, self.lightsource)
        
    def M290_forward(self, field, flip=True):
        
        field = Lens(self.f1, 0, 0, field)
        field = SmoothCircAperture(field, self.apertureRadius, self.apertureSmoothWidth)
        field = LensForvard(field, self.f2, self.focalLength+self.causticPlanes[0][1]*self.zR)
        if(flip==True):
            cfield = field.field
            cfield = np.flipud(cfield)
            cfield = np.fliplr(cfield)
            field.field = cfield
        
        return field
    
    def M290_inverse(self, field, flip=False):
        
        field = LensForvard(field,-self.f1,-1.035655*(self.focalLength+self.causticPlanes[0][1]*self.zR))
        field = Lens(-self.f1,0,0,field)
        field = CircAperture(self.apertureRadius,0,0,field)
        if(flip==True):
            cfield = field.field
            cfield = np.flipud(cfield)
            cfield = np.fliplr(cfield)
            field.field = cfield
        
        return field