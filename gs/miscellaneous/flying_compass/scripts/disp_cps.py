import h5py
import numpy as np
import matplotlib.pyplot as plt

fd = h5py.File('FOC_Record.h5', 'r+')
diff_1 = fd['/FOC/mox_diff_1'][...]
diff_2 = fd['/FOC/mox_diff_2'][...]
diff_3 = fd['/FOC/mox_diff_3'][...]
diff_4 = fd['/FOC/mox_diff_4'][...]
diff_5 = fd['/FOC/mox_diff_5'][...]
diff_6 = fd['/FOC/mox_diff_6'][...]
cp_max_1 = fd['/FOC/mox_cp_max_1'][...]
cp_max_2 = fd['/FOC/mox_cp_max_2'][...]
cp_max_3 = fd['/FOC/mox_cp_max_3'][...]
cp_max_4 = fd['/FOC/mox_cp_max_4'][...]
cp_max_5 = fd['/FOC/mox_cp_max_5'][...]
cp_max_6 = fd['/FOC/mox_cp_max_6'][...]
cp_min_1 = fd['/FOC/mox_cp_min_1'][...]
cp_min_2 = fd['/FOC/mox_cp_min_2'][...]
cp_min_3 = fd['/FOC/mox_cp_min_3'][...]
cp_min_4 = fd['/FOC/mox_cp_min_4'][...]
cp_min_5 = fd['/FOC/mox_cp_min_5'][...]
cp_min_6 = fd['/FOC/mox_cp_min_6'][...]

fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(nrows=6)

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

ax5.plot(diff_5[:,0], color='red')
ax5.plot(diff_5[:,1], color='green')
ax5.plot(diff_5[:,2], color='blue')

ax6.plot(diff_6[:,0], color='red')
ax6.plot(diff_6[:,1], color='green')
ax6.plot(diff_6[:,2], color='blue')

ax1.plot(cp_max_1[:,0], diff_1[cp_max_1[:,0],0], 'r^')
ax1.plot(cp_max_1[:,1], diff_1[cp_max_1[:,1],1], 'g^')
ax1.plot(cp_max_1[:,2], diff_1[cp_max_1[:,2],2], 'b^')

ax2.plot(cp_max_2[:,0], diff_2[cp_max_2[:,0],0], 'r^')
ax2.plot(cp_max_2[:,1], diff_2[cp_max_2[:,1],1], 'g^')
ax2.plot(cp_max_2[:,2], diff_2[cp_max_2[:,2],2], 'b^')

ax3.plot(cp_max_3[:,0], diff_3[cp_max_3[:,0],0], 'r^')
ax3.plot(cp_max_3[:,1], diff_3[cp_max_3[:,1],1], 'g^')
ax3.plot(cp_max_3[:,2], diff_3[cp_max_3[:,2],2], 'b^')

ax4.plot(cp_max_4[:,0], diff_4[cp_max_4[:,0],0], 'r^')
ax4.plot(cp_max_4[:,1], diff_4[cp_max_4[:,1],1], 'g^')
ax4.plot(cp_max_4[:,2], diff_4[cp_max_4[:,2],2], 'b^')

ax5.plot(cp_max_5[:,0], diff_5[cp_max_5[:,0],0], 'r^')
ax5.plot(cp_max_5[:,1], diff_5[cp_max_5[:,1],1], 'g^')
ax5.plot(cp_max_5[:,2], diff_5[cp_max_5[:,2],2], 'b^')

ax6.plot(cp_max_6[:,0], diff_6[cp_max_6[:,0],0], 'r^')
ax6.plot(cp_max_6[:,1], diff_6[cp_max_6[:,1],1], 'g^')
ax6.plot(cp_max_6[:,2], diff_6[cp_max_6[:,2],2], 'b^')

plt.show()
