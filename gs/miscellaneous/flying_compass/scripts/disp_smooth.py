import h5py
import numpy as np
import matplotlib.pyplot as plt

fd = h5py.File('FOC_Record.h5', 'r+')
smoothed = fd['/FOC/mox_smooth'][...]

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)
ax.plot(smoothed[:,0], color='red')
ax.plot(smoothed[:,1], color='green')
ax.plot(smoothed[:,2], color='blue')

plt.show()
