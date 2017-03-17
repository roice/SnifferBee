import h5py
import numpy as np
import math
import matplotlib.pyplot as plt

fd = h5py.File('FOC_Record.h5', 'r+')
att = fd['/FOC/attitude'][...]
pos = fd['/FOC/position'][...]

fig, (ax1, ax2) = plt.subplots(nrows=2)

ax1.plot(att[:,0]*180./np.pi, color='r', label='roll')
ax1.plot(att[:,1]*180./np.pi, color='y', label='pitch')
ax1.plot(att[:,2]*180./np.pi, color='b', label='yaw')

ax2.plot(pos[:,0], color='r', label='east')
ax2.plot(pos[:,1], color='y', label='north')
ax2.plot(pos[:,2], color='b', label='up')

plt.show()

