import h5py
import numpy as np
import matplotlib.pyplot as plt

fd = h5py.File('FOC_Record.h5', 'r+')
abs_1 = fd['/FOC/mox_abs_1'][...]
abs_2 = fd['/FOC/mox_abs_2'][...]
abs_3 = fd['/FOC/mox_abs_3'][...]
abs_4 = fd['/FOC/mox_abs_4'][...]
toa_1 = fd['/FOC/mox_toa_1'][...]
toa_2 = fd['/FOC/mox_toa_2'][...]
toa_3 = fd['/FOC/mox_toa_3'][...]
toa_4 = fd['/FOC/mox_toa_4'][...]

fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8) = plt.subplots(nrows=8)

ax1.plot(abs_1[:,0], color='r')
ax1.plot(abs_1[:,1], color='y')
ax1.plot(abs_1[:,2], color='b')

ax2.plot(abs_2[:,0], color='r')
ax2.plot(abs_2[:,1], color='y')
ax2.plot(abs_2[:,2], color='b')

ax3.plot(abs_3[:,0], color='r')
ax3.plot(abs_3[:,1], color='y')
ax3.plot(abs_3[:,2], color='b')

ax4.plot(abs_4[:,0], color='r')
ax4.plot(abs_4[:,1], color='y')
ax4.plot(abs_4[:,2], color='b')

ax5.plot(toa_1[:,0], color='r')
ax5.plot(toa_1[:,1], color='y')
ax5.plot(toa_1[:,2], color='b')

ax6.plot(toa_2[:,0], color='r')
ax6.plot(toa_2[:,1], color='y')
ax6.plot(toa_2[:,2], color='b')

ax7.plot(toa_3[:,0], color='r')
ax7.plot(toa_3[:,1], color='y')
ax7.plot(toa_3[:,2], color='b')

ax8.plot(toa_4[:,0], color='r')
ax8.plot(toa_4[:,1], color='y')
ax8.plot(toa_4[:,2], color='b')

plt.show()
