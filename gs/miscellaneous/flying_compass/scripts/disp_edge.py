import h5py
import numpy as np
import matplotlib.pyplot as plt

fd = h5py.File('FOC_Record.h5', 'r+')
diff_1 = fd['/FOC/mox_diff_1'][...][500:]
diff_2 = fd['/FOC/mox_diff_2'][...][500:]
diff_3 = fd['/FOC/mox_diff_3'][...][500:]
diff_4 = fd['/FOC/mox_diff_4'][...][500:]
edge_max_1 = fd['/FOC/mox_edge_max_1'][...][500:]
edge_max_2 = fd['/FOC/mox_edge_max_2'][...][500:]
edge_max_3 = fd['/FOC/mox_edge_max_3'][...][500:]
edge_max_4 = fd['/FOC/mox_edge_max_4'][...][500:]
edge_min_1 = fd['/FOC/mox_edge_min_1'][...][500:]
edge_min_2 = fd['/FOC/mox_edge_min_2'][...][500:]
edge_min_3 = fd['/FOC/mox_edge_min_3'][...][500:]
edge_min_4 = fd['/FOC/mox_edge_min_4'][...][500:]

fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4)

ax1.plot(diff_1[:,0], color='red')
ax1.plot(diff_1[:,1], color='green')
ax1.plot(diff_1[:,2], color='blue')

ax2.plot(diff_2[:,0], color='red')
ax2.plot(diff_2[:,1], color='green')
ax2.plot(diff_2[:,2], color='blue')

ax3.plot(diff_3[:,0], color='red')
ax3.plot(diff_3[:,1], color='green')
ax3.plot(diff_3[:,2], color='blue')

ax4.plot(diff_4[:,0], color='red')
ax4.plot(diff_4[:,1], color='green')
ax4.plot(diff_4[:,2], color='blue')

'''
ax1.plot(edge_max_1[:,0], color='red')
ax1.plot(edge_max_1[:,1], color='green')
ax1.plot(edge_max_1[:,2], color='blue')

ax2.plot(edge_max_2[:,0], color='red')
ax2.plot(edge_max_2[:,1], color='green')
ax2.plot(edge_max_2[:,2], color='blue')

ax3.plot(edge_max_3[:,0], color='red')
ax3.plot(edge_max_3[:,1], color='green')
ax3.plot(edge_max_3[:,2], color='blue')

ax4.plot(edge_max_4[:,0], color='red')
ax4.plot(edge_max_4[:,1], color='green')
ax4.plot(edge_max_4[:,2], color='blue')
'''

ax1.plot(edge_min_1[:,0], color='red')
ax1.plot(edge_min_1[:,1], color='green')
ax1.plot(edge_min_1[:,2], color='blue')

ax2.plot(edge_min_2[:,0], color='red')
ax2.plot(edge_min_2[:,1], color='green')
ax2.plot(edge_min_2[:,2], color='blue')

ax3.plot(edge_min_3[:,0], color='red')
ax3.plot(edge_min_3[:,1], color='green')
ax3.plot(edge_min_3[:,2], color='blue')

ax4.plot(edge_min_4[:,0], color='red')
ax4.plot(edge_min_4[:,1], color='green')
ax4.plot(edge_min_4[:,2], color='blue')

plt.show()
