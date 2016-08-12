import h5py
import numpy as np
import math
import matplotlib.pyplot as plt

fd = h5py.File('../data/Record_2016-07-08_15-24-37.h5', 'r+')
att = fd['att_of_robot_0'][...]
pos = fd['enu_of_robot_0'][...]

fd_foc = h5py.File('FOC_Record.h5', 'r+')
wind_dir = fd_foc['/FOC/wind_direction'][...]

print np.asarray(wind_dir[:,0]).shape

xy_dir = []
xy_str = []
for i in range(len(wind_dir)):
    direct = math.atan(wind_dir[i,1]/wind_dir[i,0])
    strength = wind_dir[i,1]*wind_dir[i,1] + wind_dir[i,0]*wind_dir[i,0]
    xy_dir.append(direct)
    xy_str.append(strength)

att = att*180./np.pi

fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4)

ax1.plot(att[:,0], color='r', label='roll')
ax1.plot(att[:,1], color='y', label='pitch')
ax1.plot(att[:,2], color='b', label='yaw')

ax2.plot(pos[:,0], color='r', label='east')
ax2.plot(pos[:,1], color='y', label='north')
ax2.plot(pos[:,2], color='b', label='up')

ax3.plot(xy_dir)

ax4.plot(xy_str)

plt.show()
