import h5py
import numpy as np
import matplotlib.pyplot as plt

fd = h5py.File('FOC_Record.h5', 'r+')
std = fd['/FOC/mox_std'][...]


fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)

ax.plot(std[:,0], color='r')
ax.plot(std[:,1], color='g')
ax.plot(std[:,2], color='b')

plt.show()
