import h5py
import numpy as np
import math
import matplotlib.pyplot as plt

fd = h5py.File('FOC_Record.h5', 'r+')
fd_ori = h5py.File('../data/Record_2016-08-03_17-30-06.h5')
wind = fd['FOC/wind'][...]
att = fd_ori['robot1/att'][...]
wind_filtered = fd['FOC/wind_filtered'][...]

direction = []
for i in range(len(wind)):
    if (wind[i,1] != 0.):
        direction.append(-math.atan2(-wind[i,0], -wind[i,1]))
    else:
        if (wind[i,0] > 0):
            direction.append(np.pi/2.)
        else:
            direction.append(-np.pi/2.)

direction_filtered = []
for i in range(len(wind_filtered)):
    if (wind_filtered[i,1] != 0.):
        direction_filtered.append(-math.atan2(-wind_filtered[i,0], -wind_filtered[i,1]))
    else:
        if (wind_filtered[i,0] > 0):
            direction_filtered.append(np.pi/2.)
        else:
            direction_filtered.append(-np.pi/2.)

fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(nrows=6)

ax1.plot(wind[:,0], color='red')
ax1.plot(wind[:,1], color='green')
ax1.plot(wind[:,2], color='blue')

ax2.plot(wind_filtered[:,0], color='red')
ax2.plot(wind_filtered[:,1], color='green')
ax2.plot(wind_filtered[:,2], color='blue')

ax3.plot(np.asarray(direction)*180./np.pi)

ax4.plot(direction_filtered)

ax5.plot(att[0:622,2]*180./np.pi)

ax6.plot((np.asarray(direction)+att[0:622,2])*180./np.pi)

plt.show()
