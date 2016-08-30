import h5py
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter

fd = h5py.File('FOC_Record.h5', 'r+')
direction = fd['/FOC/est_direction'][...]
clustering = fd['/FOC/est_clustering'][...]
std = fd['/FOC/est_std'][...]
wind_xy = fd['/FOC/est_wind_p'][...]
wind = fd['/FOC/est_wind'][...]

d_x = direction[:,0]
d_y = direction[:,1]

theta = []
for i in range(len(direction)):
    if clustering[i] < 0:
        continue
    ang = math.atan2(-d_x[i], d_y[i])
    theta.append(ang)
theta = np.asarray(theta)
#theta = gaussian_filter(np.asarray(theta), sigma=100)

direction = []
for i in range(len(wind_xy)):
    direction.append(math.atan2(-wind_xy[i,0], wind_xy[i,1]))
direction = np.asarray(direction)

#theta = gaussian_filter(theta, sigma=50)
direction = gaussian_filter(direction, sigma=50)

s = gaussian_filter(theta, sigma=50)

wind_direct = []
for i in range(len(wind)):
    wind_direct.append(math.atan2(-wind[i,0], wind[i,1]))
wind_direct = np.asarray(wind_direct)

fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4)

ax1.plot(theta*180./np.pi, color = 'r')
ax1.plot(wind_direct*180./np.pi, color = 'g')

ax2.plot(clustering)
ax3.plot(std)
ax4.plot(s*180./np.pi)

plt.show()
