import h5py
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter

fd = h5py.File('FOC_Record.h5', 'r+')
direction = fd['/FOC/est_direction'][...]
belief = fd['/FOC/est_belief'][...]
wind_xy = fd['/FOC/est_wind_speed_xy'][...]

d_x = direction[:,0]
d_y = direction[:,1]

theta = []
for i in range(len(direction)):
    if belief[i] < 0.3:
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

fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4)

ax1.plot(theta*180./np.pi)

ax2.plot(belief)
ax3.plot(direction)
ax4.plot(s*180./np.pi)

plt.show()
