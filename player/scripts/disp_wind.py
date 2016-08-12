import h5py
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter

fd = h5py.File('../data/Record_2016-08-03_17-30-06.h5', 'r+')
wind = fd['robot1/wind'][...]

direction = []
for i in range(len(wind)):
    if (wind[i,1] != 0.):
        direction.append(math.atan2(-wind[i,0], -wind[i,1]))
    else:
        if (wind[i,0] > 0):
            direction.append(-np.pi/2.)
        else:
            direction.append(np.pi/2.)

#direction = gaussian_filter(direction, sigma = 50)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)
'''
ax.plot(wind[:,0], color='red')
ax.plot(wind[:,1], color='yellow')
ax.plot(wind[:,2], color='blue')
'''
ax.plot(np.asarray(direction)*180./np.pi)
plt.show()
