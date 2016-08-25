import h5py
import numpy as np
import matplotlib.pyplot as plt

fd = h5py.File('FOC_Record.h5', 'r+')
sm_1 = fd['/FOC/mox_smooth_1'][...]
sm_2 = fd['/FOC/mox_smooth_2'][...]
sm_3 = fd['/FOC/mox_smooth_3'][...]

fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(nrows=6)

ax1.plot(sm_1[:,0], color='red')
ax1.plot(sm_1[:,1], color='green')
ax1.plot(sm_1[:,2], color='blue')

ax2.plot(sm_2[:,0], color='red')
ax2.plot(sm_2[:,1], color='green')
ax2.plot(sm_2[:,2], color='blue')

ax3.plot(sm_3[:,0], color='red')
ax3.plot(sm_3[:,1], color='green')
ax3.plot(sm_3[:,2], color='blue')






plt.show()
