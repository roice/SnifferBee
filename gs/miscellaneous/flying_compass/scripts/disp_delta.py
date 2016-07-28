import h5py
import numpy as np
import matplotlib.pyplot as plt

fd = h5py.File('FOC_Record.h5', 'r+')
std = fd['/FOC/mox_std'][...]
toa = fd['/FOC/mox_toa'][...]

fig, (ax1, ax2) = plt.subplots(nrows=2)

ax1.plot(std[:,0], color='r')
ax1.plot(std[:,1], color='y')
ax1.plot(std[:,2], color='b')
ax2.plot(toa[:,0], color='r')
ax2.plot(toa[:,1], color='y')
ax2.plot(toa[:,2], color='b')

plt.show()
