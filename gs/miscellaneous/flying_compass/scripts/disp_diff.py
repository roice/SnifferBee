import h5py
import numpy as np
import matplotlib.pyplot as plt

fd = h5py.File('FOC_Record.h5', 'r+')
diff_1 = fd['/FOC/mox_diff_1'][...]
diff_2 = fd['/FOC/mox_diff_2'][...]
diff_3 = fd['/FOC/mox_diff_3'][...]

'''
diff_4 = fd['/FOC/mox_diff_4'][...][500:]
diff_5 = fd['/FOC/mox_diff_5'][...][500:]
diff_6 = fd['/FOC/mox_diff_6'][...][500:]
'''


'''
diff = diff[2000:6000,:]

diff_f = diff[:,0]/np.std(diff[:,0])
diff_l = diff[:,1]/np.std(diff[:,1])
diff_r = diff[:,2]/np.std(diff[:,2])
'''


fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(nrows=6)

ax1.plot(diff_1[:,0], color='r')
ax1.plot(diff_1[:,1], color='g')
ax1.plot(diff_1[:,2], color='b')

ax2.plot(diff_2[:,0], color='r')
ax2.plot(diff_2[:,1], color='g')
ax2.plot(diff_2[:,2], color='b')

ax3.plot(diff_3[:,0], color='r')
ax3.plot(diff_3[:,1], color='g')
ax3.plot(diff_3[:,2], color='b')

'''
ax4.plot(diff_4[:,0], color='r')
ax4.plot(diff_4[:,1], color='g')
ax4.plot(diff_4[:,2], color='b')

ax5.plot(diff_5[:,0], color='r')
ax5.plot(diff_5[:,1], color='g')
ax5.plot(diff_5[:,2], color='b')

ax6.plot(diff_6[:,0], color='r')
ax6.plot(diff_6[:,1], color='g')
ax6.plot(diff_6[:,2], color='b')
'''

plt.show()
