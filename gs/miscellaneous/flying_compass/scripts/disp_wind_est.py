import h5py
import numpy as np
import math
import matplotlib.pyplot as plt

fd = h5py.File('FOC_Record.h5', 'r+')
wind_speed_xy = fd['/FOC/est_wind_speed_xy'][...]
wind_speed_en = fd['/FOC/est_wind_speed_en'][...]
wind_speed_filtered_xy = fd['/FOC/est_wind_speed_filtered_xy'][...]
valid = fd['/FOC/est_valid'][...]

fd_ori = h5py.File('../data/Record_2016-06-07_22-30-51.h5', 'r+')
att = fd_ori['att_of_robot_0'][...]
pos = fd_ori['enu_of_robot_0'][...]

direction = []
for i in range(len(valid)):
    if valid[i]:
        if (wind_speed_xy[i,1] != 0.):
            direction.append(math.atan2(-wind_speed_xy[i,0], -wind_speed_xy[i,1]))
        else:
            if (wind_speed_xy[i,0] > 0):
                direction.append(-np.pi/2.)
            else:
                direction.append(np.pi/2.)
    else:
        direction.append(3*np.pi)

direction_filtered = []
for i in range(len(valid)):
    if valid[i]:
        if (wind_speed_filtered_xy[i,1] != 0.):
            direction_filtered.append(math.atan2(-wind_speed_filtered_xy[i,0], -wind_speed_filtered_xy[i,1]))
        else:
            if (wind_speed_filtered_xy[i,0] > 0):
                direction_filtered.append(-np.pi/2.)
            else:
                direction_filtered.append(np.pi/2.)
    else:
        direction_filtered.append(3*np.pi)

for i in range(len(direction)):
    direction[i] += -att[i,2]

fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4)

ax1.plot(wind_speed_xy[:,0], color='red')
ax1.plot(wind_speed_xy[:,1], color='blue')

#ax2.plot(wind_speed_filtered_xy[:,0], color='red')
#ax2.plot(wind_speed_filtered_xy[:,1], color='blue')

ax2.plot(att[:,2]*180./np.pi)

ax3.plot(np.asarray(direction)*180./np.pi)

ax4.plot(pos[:,1])

plt.show()
