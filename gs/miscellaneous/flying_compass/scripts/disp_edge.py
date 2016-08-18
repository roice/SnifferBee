import h5py
import numpy as np
import matplotlib.pyplot as plt
from pylab import *

fd = h5py.File('FOC_Record.h5', 'r+')
grad = fd['/FOC/mox_gradient'][...]
edge_max = fd['/FOC/mox_edge_max'][...]
edge_min = fd['/FOC/mox_edge_min'][...]
cp_max = fd['/FOC/mox_cp_max'][...]
cp_min = fd['/FOC/mox_cp_min'][...]
att = h5py.File('../data/Record_2016-08-03_17-30-06.h5', 'r+')['robot1/att'][...]

print len(cp_max)
print len(cp_min)

set_printoptions(threshold='nan')
print cp_max

fig, (ax1, ax2, ax3) = plt.subplots(nrows=3)

ax1.plot(grad[200:,0], color='red')
ax1.plot(grad[200:,1], color='green')
ax1.plot(grad[200:,2], color='blue')

ax2.plot(edge_min[200:,0], color='red')
ax2.plot(edge_min[200:,1], color='green')
ax2.plot(edge_min[200:,2], color='blue')

#ax3.stem(cp_max[:,0], edge_max[cp_max[:,0]])

#ax2.plot(att[:,2]*180./np.pi)

plt.show()
