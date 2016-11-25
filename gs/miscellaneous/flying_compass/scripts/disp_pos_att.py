import h5py
import numpy as np
import math
import matplotlib.pyplot as plt

fd = h5py.File('../data/Record_2016-10-27_20-16-05.h5', 'r+')
att = fd['robot1/att'][...]
pos = fd['robot1/enu'][...]

fig, (ax1, ax2) = plt.subplots(nrows=2)

ax1.plot(att[:,0]*180./np.pi, color='r', label='roll')
ax1.plot(att[:,1]*180./np.pi, color='y', label='pitch')
ax1.plot(att[:,2]*180./np.pi, color='b', label='yaw')

ax2.plot(pos[:,0], color='r', label='east')
ax2.plot(pos[:,1], color='y', label='north')
ax2.plot(pos[:,2], color='b', label='up')

plt.show()

