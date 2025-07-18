import numpy as np

chair = np.load('chair/runtime.npy')
rec = np.load('rec/runtime.npy')
gaussian = np.load('gaussian/runtime.npy')
tear = np.load('tear/runtime.npy')
ring = np.load('ring/runtime.npy')
hat = np.load('hat/runtime.npy')

chair = np.mean(chair)
rec = np.mean(rec)
gaussian = np.mean(gaussian)
tear = np.mean(tear)
ring = np.mean(ring)
hat = np.mean(hat)

t = chair+rec+gaussian+tear+ring+hat
t = t/6

print(str(t)+' seconds')