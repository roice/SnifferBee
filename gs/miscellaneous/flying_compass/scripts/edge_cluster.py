import h5py
import numpy as np
import matplotlib.pyplot as plt

fd = h5py.File('FOC_Record.h5', 'r+')
grad = fd['/FOC/mox_gradient'][...]
edge_max = fd['/FOC/mox_edge_max'][...]
edge_min = fd['/FOC/mox_edge_min'][...]
att = h5py.File('../data/Record_2016-08-03_17-30-06.h5', 'r+')['robot1/att'][...]

fig, (ax1, ax2) = plt.subplots(nrows=2)

ax1.plot(grad[:,0], color='red')
ax1.plot(grad[:,1], color='green')
ax1.plot(grad[:,2], color='blue')
ax1.plot(edge_max[:,0], color='red')
ax1.plot(edge_max[:,1], color='green')
ax1.plot(edge_max[:,2], color='blue')

ax2.plot(att[:,2]*180./np.pi)

plt.show()
