import h5py
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter

fd = h5py.File('FOC_Record.h5', 'r+')
wind = fd['FOC/wind_filtered'][...]
direction = fd['FOC/direction'][...]
valid = fd['FOC/est_valid'][...]

direction[:,0] = gaussian_filter(direction[:,0], sigma = 10)
direction[:,1] = gaussian_filter(direction[:,1], sigma = 10)

est_direction = []
for i in range(len(valid)):
    if valid[i]:
        if (direction[i,1] != 0.):
            est_direction.append(math.atan2(-direction[i,0], -direction[i,1]))
        else:
            if (direction[i,0] > 0):
                est_direction.append(-np.pi/2.)
            else:
                est_direction.append(np.pi/2.)
    else:
        est_direction.append(3*np.pi)

est_wind = []
for i in range(len(valid)):
    if valid[i]:
        if (wind[i,1] != 0.):
            est_wind.append(math.atan2(-wind[i,0], -wind[i,1]))
        else:
            if (wind[i,0] > 0):
                est_wind.append(-np.pi/2.)
            else:
                est_wind.append(np.pi/2.)
    else:
        est_wind.append(3*np.pi)

angle = []
for i in range(len(valid)):
    if valid[i]:
        if wind[i,0] == 0 and wind[i,1] == 0:
            theta = 3*np.pi
        else:
            theta = math.acos((wind[i,0]*direction[i,0]+wind[i,1]*direction[i,1])/(math.sqrt((wind[i,0]*wind[i,0]+wind[i,1]*wind[i,1])*(direction[i,0]*direction[i,0]+direction[i,1]*direction[i,1]))))
        angle.append(theta)
    else:
        angle.append(3*np.pi)

wind_strength = []
for i in range(len(wind)):
    strength = wind[i,0]*wind[i,0]+wind[i,1]*wind[i,1]
    wind_strength.append(strength)

fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4)

ax1.plot(est_direction, color='red');
ax1.plot(est_wind, color='blue')

ax2.plot(angle)

ax3.plot(wind_strength)

plt.show()


