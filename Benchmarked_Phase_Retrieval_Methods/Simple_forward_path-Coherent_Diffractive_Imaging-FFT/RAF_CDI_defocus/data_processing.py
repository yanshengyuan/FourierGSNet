import numpy as np
import h5py
import matplotlib.pyplot as plt

defocus_mask = np.load('Mask_defocus_200.npy')
phase_mask = np.angle(defocus_mask)
angular_mask = np.exp(1j * phase_mask)
plt.imsave("Defocus_phase_mask.png", np.angle(defocus_mask), cmap='gray')

'''
with h5py.File('train_clean.h5', 'r') as f:
    for i in range(len(f)-1):
        sample_id = f"{i:06d}"
        
        real = f[sample_id][0]
        imag = f[sample_id][1]
        np.save("train/real/npy/"+sample_id+".npy", real)
        plt.imsave("train/real/img/"+sample_id+".png", real, cmap='gray')
        np.save("train/imag/npy/"+sample_id+".npy", imag)
        plt.imsave("train/imag/img/"+sample_id+".png", imag, cmap='gray')
        
        phase = np.arctan2(imag, real)
        lightsource = real**2 + imag**2
        lightsource = (lightsource - np.min(lightsource)) / (np.max(lightsource) - np.min(lightsource)) * 255
        
        np.save("train/phase/npy/"+sample_id+".npy", phase)
        plt.imsave("train/phase/img/"+sample_id+".png", phase, cmap='gray')
        np.save("train/lightsource/npy/"+sample_id+".npy", lightsource)
        plt.imsave("train/lightsource/img/"+sample_id+".png", lightsource, cmap='gray')
    
        lightfield = real + imag*1j
        lightfield = lightfield * angular_mask
        
        FFT = np.fft.fft2(lightfield)
        FFT = np.fft.fftshift(FFT)
        
        FFT_intensity = np.abs(FFT)**2
        FFT_intensity = (FFT_intensity - np.min(FFT_intensity)) / (np.max(FFT_intensity) - np.min(FFT_intensity)) * 255
        #np.save("train/intensity/npy/"+sample_id+".npy", FFT_intensity)
        #plt.imsave("train/intensity/img/"+sample_id+".png", FFT_intensity, cmap='gray')
        print(sample_id)

with h5py.File('test_clean.h5', 'r') as f:
    for i in range(len(f)):
        sample_id = f"{i:06d}"
        
        real = f[sample_id][0]
        imag = f[sample_id][1]
        np.save("test/real/npy/"+sample_id+".npy", real)
        plt.imsave("test/real/img/"+sample_id+".png", real, cmap='gray')
        np.save("test/imag/npy/"+sample_id+".npy", imag)
        plt.imsave("test/imag/img/"+sample_id+".png", imag, cmap='gray')
        
        phase = np.arctan2(imag, real)
        lightsource = real**2 + imag**2
        lightsource = (lightsource - np.min(lightsource)) / (np.max(lightsource) - np.min(lightsource)) * 255
        
        np.save("test/phase/npy/"+sample_id+".npy", phase)
        plt.imsave("test/phase/img/"+sample_id+".png", phase, cmap='gray')
        np.save("test/lightsource/npy/"+sample_id+".npy", lightsource)
        plt.imsave("test/lightsource/img/"+sample_id+".png", lightsource, cmap='gray')
    
        lightfield = real + imag*1j
        lightfield = lightfield * angular_mask
        
        FFT = np.fft.fft2(lightfield)
        FFT = np.fft.fftshift(FFT)
        
        FFT_intensity = np.abs(FFT)**2
        #np.save("test/intensity/npy/"+sample_id+".npy", FFT_intensity)
        #plt.imsave("test/intensity/img/"+sample_id+".png", FFT_intensity, cmap='gray')
        print(sample_id)
'''

#'''
with h5py.File('train_intensity_full.h5', 'r') as f:
    for i in range(len(f)-1):
        sample_id = f"{i:06d}"
        
        measurement = f[sample_id]
        
        np.save("train/intensity/npy/"+sample_id+".npy", measurement)
        plt.imsave("train/intensity/img/"+sample_id+".png", measurement, cmap='gray')
        
        print(sample_id)
#'''

#''' 
with h5py.File('test_intensity_full.h5', 'r') as f:
    for i in range(len(f)):
        sample_id = f"{i:06d}"
        
        measurement = f[sample_id]
        
        np.save("test/intensity/npy/"+sample_id+".npy", measurement)
        plt.imsave("test/intensity/img/"+sample_id+".png", measurement, cmap='gray')
        
        print(sample_id)
#'''