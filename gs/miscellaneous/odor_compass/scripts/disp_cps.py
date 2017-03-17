import h5py
import numpy as np
import matplotlib.pyplot as plt

fd = h5py.File('FOC_Record.h5', 'r+')
diff_1_1 = fd['/FOC/mox_diff_group_1_layer_1'][...]
diff_1_2 = fd['/FOC/mox_diff_group_1_layer_2'][...]
diff_1_3 = fd['/FOC/mox_diff_group_1_layer_3'][...]
#diff_1_4 = fd['/FOC/mox_diff_group_1_layer_4'][...]
diff_2_1 = fd['/FOC/mox_diff_group_2_layer_1'][...]
diff_2_2 = fd['/FOC/mox_diff_group_2_layer_2'][...]
diff_2_3 = fd['/FOC/mox_diff_group_2_layer_3'][...]
#diff_2_4 = fd['/FOC/mox_diff_group_2_layer_4'][...]
diff_3_1 = fd['/FOC/mox_diff_group_3_layer_1'][...]
diff_3_2 = fd['/FOC/mox_diff_group_3_layer_2'][...]
diff_3_3 = fd['/FOC/mox_diff_group_3_layer_3'][...]
#diff_3_4 = fd['/FOC/mox_diff_group_3_layer_4'][...]
diff_4_1 = fd['/FOC/mox_diff_group_4_layer_1'][...]
diff_4_2 = fd['/FOC/mox_diff_group_4_layer_2'][...]
diff_4_3 = fd['/FOC/mox_diff_group_4_layer_3'][...]
#diff_4_4 = fd['/FOC/mox_diff_group_4_layer_4'][...]
diff_5_1 = fd['/FOC/mox_diff_group_5_layer_1'][...]
diff_5_2 = fd['/FOC/mox_diff_group_5_layer_2'][...]
diff_5_3 = fd['/FOC/mox_diff_group_5_layer_3'][...]
#diff_5_4 = fd['/FOC/mox_diff_group_5_layer_4'][...]
diff_6_1 = fd['/FOC/mox_diff_group_6_layer_1'][...]
diff_6_2 = fd['/FOC/mox_diff_group_6_layer_2'][...]
diff_6_3 = fd['/FOC/mox_diff_group_6_layer_3'][...]
#diff_6_4 = fd['/FOC/mox_diff_group_6_layer_4'][...]

cp_max_1_1 = fd['/FOC/mox_cp_max_group_1_layer_1'][...]
cp_max_1_2 = fd['/FOC/mox_cp_max_group_1_layer_2'][...]
cp_max_1_3 = fd['/FOC/mox_cp_max_group_1_layer_3'][...]
cp_max_2_1 = fd['/FOC/mox_cp_max_group_2_layer_1'][...]
cp_max_2_2 = fd['/FOC/mox_cp_max_group_2_layer_2'][...]
cp_max_2_3 = fd['/FOC/mox_cp_max_group_2_layer_3'][...]
cp_max_3_1 = fd['/FOC/mox_cp_max_group_3_layer_1'][...]
cp_max_3_2 = fd['/FOC/mox_cp_max_group_3_layer_2'][...]
cp_max_3_3 = fd['/FOC/mox_cp_max_group_3_layer_3'][...]
cp_max_4_1 = fd['/FOC/mox_cp_max_group_4_layer_1'][...]
cp_max_4_2 = fd['/FOC/mox_cp_max_group_4_layer_2'][...]
cp_max_4_3 = fd['/FOC/mox_cp_max_group_4_layer_3'][...]
cp_max_5_1 = fd['/FOC/mox_cp_max_group_5_layer_1'][...]
cp_max_5_2 = fd['/FOC/mox_cp_max_group_5_layer_2'][...]
cp_max_5_3 = fd['/FOC/mox_cp_max_group_5_layer_3'][...]
cp_max_6_1 = fd['/FOC/mox_cp_max_group_6_layer_1'][...]
cp_max_6_2 = fd['/FOC/mox_cp_max_group_6_layer_2'][...]
cp_max_6_3 = fd['/FOC/mox_cp_max_group_6_layer_3'][...]

cp_min_1_1 = fd['/FOC/mox_cp_min_group_1_layer_1'][...]
cp_min_1_2 = fd['/FOC/mox_cp_min_group_1_layer_2'][...]
cp_min_1_3 = fd['/FOC/mox_cp_min_group_1_layer_3'][...]
cp_min_2_1 = fd['/FOC/mox_cp_min_group_2_layer_1'][...]
cp_min_2_2 = fd['/FOC/mox_cp_min_group_2_layer_2'][...]
cp_min_2_3 = fd['/FOC/mox_cp_min_group_2_layer_3'][...]
cp_min_3_1 = fd['/FOC/mox_cp_min_group_3_layer_1'][...]
cp_min_3_2 = fd['/FOC/mox_cp_min_group_3_layer_2'][...]
cp_min_3_3 = fd['/FOC/mox_cp_min_group_3_layer_3'][...]
cp_min_4_1 = fd['/FOC/mox_cp_min_group_4_layer_1'][...]
cp_min_4_2 = fd['/FOC/mox_cp_min_group_4_layer_2'][...]
cp_min_4_3 = fd['/FOC/mox_cp_min_group_4_layer_3'][...]
cp_min_5_1 = fd['/FOC/mox_cp_min_group_5_layer_1'][...]
cp_min_5_2 = fd['/FOC/mox_cp_min_group_5_layer_2'][...]
cp_min_5_3 = fd['/FOC/mox_cp_min_group_5_layer_3'][...]
cp_min_6_1 = fd['/FOC/mox_cp_min_group_6_layer_1'][...]
cp_min_6_2 = fd['/FOC/mox_cp_min_group_6_layer_2'][...]
cp_min_6_3 = fd['/FOC/mox_cp_min_group_6_layer_3'][...]

cp_max_begin_1_1 = np.asarray([ np.min(cp_max_1_1[i,:]) for i in range(len(cp_max_1_1)) ])
cp_max_begin_1_2 = np.asarray([ np.min(cp_max_1_2[i,:]) for i in range(len(cp_max_1_2)) ])
cp_max_begin_1_3 = np.asarray([ np.min(cp_max_1_3[i,:]) for i in range(len(cp_max_1_3)) ])
cp_max_begin_2_1 = np.asarray([ np.min(cp_max_2_1[i,:]) for i in range(len(cp_max_2_1)) ])
cp_max_begin_2_2 = np.asarray([ np.min(cp_max_2_2[i,:]) for i in range(len(cp_max_2_2)) ])
cp_max_begin_2_3 = np.asarray([ np.min(cp_max_2_3[i,:]) for i in range(len(cp_max_2_3)) ])
cp_max_begin_3_1 = np.asarray([ np.min(cp_max_3_1[i,:]) for i in range(len(cp_max_3_1)) ])
cp_max_begin_3_2 = np.asarray([ np.min(cp_max_3_2[i,:]) for i in range(len(cp_max_3_2)) ])
cp_max_begin_3_3 = np.asarray([ np.min(cp_max_3_3[i,:]) for i in range(len(cp_max_3_3)) ])
cp_max_begin_4_1 = np.asarray([ np.min(cp_max_4_1[i,:]) for i in range(len(cp_max_4_1)) ])
cp_max_begin_4_2 = np.asarray([ np.min(cp_max_4_2[i,:]) for i in range(len(cp_max_4_2)) ])
cp_max_begin_4_3 = np.asarray([ np.min(cp_max_4_3[i,:]) for i in range(len(cp_max_4_3)) ])
cp_max_begin_5_1 = np.asarray([ np.min(cp_max_5_1[i,:]) for i in range(len(cp_max_5_1)) ])
cp_max_begin_5_2 = np.asarray([ np.min(cp_max_5_2[i,:]) for i in range(len(cp_max_5_2)) ])
cp_max_begin_5_3 = np.asarray([ np.min(cp_max_5_3[i,:]) for i in range(len(cp_max_5_3)) ])
cp_max_begin_6_1 = np.asarray([ np.min(cp_max_6_1[i,:]) for i in range(len(cp_max_6_1)) ])
cp_max_begin_6_2 = np.asarray([ np.min(cp_max_6_2[i,:]) for i in range(len(cp_max_6_2)) ])
cp_max_begin_6_3 = np.asarray([ np.min(cp_max_6_3[i,:]) for i in range(len(cp_max_6_3)) ])

cp_min_begin_1_1 = np.asarray([ np.min(cp_min_1_1[i,:]) for i in range(len(cp_min_1_1)) ])
cp_min_begin_1_2 = np.asarray([ np.min(cp_min_1_2[i,:]) for i in range(len(cp_min_1_2)) ])
cp_min_begin_1_3 = np.asarray([ np.min(cp_min_1_3[i,:]) for i in range(len(cp_min_1_3)) ])
cp_min_begin_2_1 = np.asarray([ np.min(cp_min_2_1[i,:]) for i in range(len(cp_min_2_1)) ])
cp_min_begin_2_2 = np.asarray([ np.min(cp_min_2_2[i,:]) for i in range(len(cp_min_2_2)) ])
cp_min_begin_2_3 = np.asarray([ np.min(cp_min_2_3[i,:]) for i in range(len(cp_min_2_3)) ])
cp_min_begin_3_1 = np.asarray([ np.min(cp_min_3_1[i,:]) for i in range(len(cp_min_3_1)) ])
cp_min_begin_3_2 = np.asarray([ np.min(cp_min_3_2[i,:]) for i in range(len(cp_min_3_2)) ])
cp_min_begin_3_3 = np.asarray([ np.min(cp_min_3_3[i,:]) for i in range(len(cp_min_3_3)) ])
cp_min_begin_4_1 = np.asarray([ np.min(cp_min_4_1[i,:]) for i in range(len(cp_min_4_1)) ])
cp_min_begin_4_2 = np.asarray([ np.min(cp_min_4_2[i,:]) for i in range(len(cp_min_4_2)) ])
cp_min_begin_4_3 = np.asarray([ np.min(cp_min_4_3[i,:]) for i in range(len(cp_min_4_3)) ])
cp_min_begin_5_1 = np.asarray([ np.min(cp_min_5_1[i,:]) for i in range(len(cp_min_5_1)) ])
cp_min_begin_5_2 = np.asarray([ np.min(cp_min_5_2[i,:]) for i in range(len(cp_min_5_2)) ])
cp_min_begin_5_3 = np.asarray([ np.min(cp_min_5_3[i,:]) for i in range(len(cp_min_5_3)) ])
cp_min_begin_6_1 = np.asarray([ np.min(cp_min_6_1[i,:]) for i in range(len(cp_min_6_1)) ])
cp_min_begin_6_2 = np.asarray([ np.min(cp_min_6_2[i,:]) for i in range(len(cp_min_6_2)) ])
cp_min_begin_6_3 = np.asarray([ np.min(cp_min_6_3[i,:]) for i in range(len(cp_min_6_3)) ])

fig, axes = plt.subplots(nrows=6, ncols=3, figsize=(6, 6))

axes[0,0].plot(diff_1_1[:,0], color='r')
axes[0,0].plot(diff_1_1[:,1], color='g')
axes[0,0].plot(diff_1_1[:,2], color='b')

axes[0,1].plot(diff_1_2[:,0], color='r')
axes[0,1].plot(diff_1_2[:,1], color='g')
axes[0,1].plot(diff_1_2[:,2], color='b')

axes[0,2].plot(diff_1_3[:,0], color='r')
axes[0,2].plot(diff_1_3[:,1], color='g')
axes[0,2].plot(diff_1_3[:,2], color='b')

axes[1,0].plot(diff_2_1[:,0], color='r')
axes[1,0].plot(diff_2_1[:,1], color='g')
axes[1,0].plot(diff_2_1[:,2], color='b')

axes[1,1].plot(diff_2_2[:,0], color='r')
axes[1,1].plot(diff_2_2[:,1], color='g')
axes[1,1].plot(diff_2_2[:,2], color='b')

axes[1,2].plot(diff_2_3[:,0], color='r')
axes[1,2].plot(diff_2_3[:,1], color='g')
axes[1,2].plot(diff_2_3[:,2], color='b')

axes[2,0].plot(diff_3_1[:,0], color='r')
axes[2,0].plot(diff_3_1[:,1], color='g')
axes[2,0].plot(diff_3_1[:,2], color='b')

axes[2,1].plot(diff_3_2[:,0], color='r')
axes[2,1].plot(diff_3_2[:,1], color='g')
axes[2,1].plot(diff_3_2[:,2], color='b')

axes[2,2].plot(diff_3_3[:,0], color='r')
axes[2,2].plot(diff_3_3[:,1], color='g')
axes[2,2].plot(diff_3_3[:,2], color='b')

axes[3,0].plot(diff_4_1[:,0], color='r')
axes[3,0].plot(diff_4_1[:,1], color='g')
axes[3,0].plot(diff_4_1[:,2], color='b')

axes[3,1].plot(diff_4_2[:,0], color='r')
axes[3,1].plot(diff_4_2[:,1], color='g')
axes[3,1].plot(diff_4_2[:,2], color='b')

axes[3,2].plot(diff_4_3[:,0], color='r')
axes[3,2].plot(diff_4_3[:,1], color='g')
axes[3,2].plot(diff_4_3[:,2], color='b')

axes[4,0].plot(diff_5_1[:,0], color='r')
axes[4,0].plot(diff_5_1[:,1], color='g')
axes[4,0].plot(diff_5_1[:,2], color='b')

axes[4,1].plot(diff_5_2[:,0], color='r')
axes[4,1].plot(diff_5_2[:,1], color='g')
axes[4,1].plot(diff_5_2[:,2], color='b')

axes[4,2].plot(diff_5_3[:,0], color='r')
axes[4,2].plot(diff_5_3[:,1], color='g')
axes[4,2].plot(diff_5_3[:,2], color='b')

axes[5,0].plot(diff_6_1[:,0], color='r')
axes[5,0].plot(diff_6_1[:,1], color='g')
axes[5,0].plot(diff_6_1[:,2], color='b')

axes[5,1].plot(diff_6_2[:,0], color='r')
axes[5,1].plot(diff_6_2[:,1], color='g')
axes[5,1].plot(diff_6_2[:,2], color='b')

axes[5,2].plot(diff_6_3[:,0], color='r')
axes[5,2].plot(diff_6_3[:,1], color='g')
axes[5,2].plot(diff_6_3[:,2], color='b')

axes[0,0].plot(cp_max_1_1[:,0], diff_1_1[cp_max_1_1[:,0],0], 'r^')
axes[0,0].plot(cp_max_1_1[:,1], diff_1_1[cp_max_1_1[:,1],1], 'g^')
axes[0,0].plot(cp_max_1_1[:,2], diff_1_1[cp_max_1_1[:,2],2], 'b^')
axes[0,0].plot(cp_max_begin_1_1, np.zeros_like(cp_max_begin_1_1), 'ko')

axes[0,1].plot(cp_max_1_2[:,0], diff_1_2[cp_max_1_2[:,0],0], 'r^')
axes[0,1].plot(cp_max_1_2[:,1], diff_1_2[cp_max_1_2[:,1],1], 'g^')
axes[0,1].plot(cp_max_1_2[:,2], diff_1_2[cp_max_1_2[:,2],2], 'b^')
axes[0,1].plot(cp_max_begin_1_2, np.zeros_like(cp_max_begin_1_2), 'ko')

axes[0,2].plot(cp_max_1_3[:,0], diff_1_3[cp_max_1_3[:,0],0], 'r^')
axes[0,2].plot(cp_max_1_3[:,1], diff_1_3[cp_max_1_3[:,1],1], 'g^')
axes[0,2].plot(cp_max_1_3[:,2], diff_1_3[cp_max_1_3[:,2],2], 'b^')
axes[0,2].plot(cp_max_begin_1_3, np.zeros_like(cp_max_begin_1_3), 'ko')

axes[1,0].plot(cp_max_2_1[:,0], diff_2_1[cp_max_2_1[:,0],0], 'r^')
axes[1,0].plot(cp_max_2_1[:,1], diff_2_1[cp_max_2_1[:,1],1], 'g^')
axes[1,0].plot(cp_max_2_1[:,2], diff_2_1[cp_max_2_1[:,2],2], 'b^')
axes[1,0].plot(cp_max_begin_2_1, np.zeros_like(cp_max_begin_2_1), 'ko')

axes[1,1].plot(cp_max_2_2[:,0], diff_2_2[cp_max_2_2[:,0],0], 'r^')
axes[1,1].plot(cp_max_2_2[:,1], diff_2_2[cp_max_2_2[:,1],1], 'g^')
axes[1,1].plot(cp_max_2_2[:,2], diff_2_2[cp_max_2_2[:,2],2], 'b^')
axes[1,1].plot(cp_max_begin_2_2, np.zeros_like(cp_max_begin_2_2), 'ko')

axes[1,2].plot(cp_max_2_3[:,0], diff_2_3[cp_max_2_3[:,0],0], 'r^')
axes[1,2].plot(cp_max_2_3[:,1], diff_2_3[cp_max_2_3[:,1],1], 'g^')
axes[1,2].plot(cp_max_2_3[:,2], diff_2_3[cp_max_2_3[:,2],2], 'b^')
axes[1,2].plot(cp_max_begin_2_3, np.zeros_like(cp_max_begin_2_3), 'ko')

axes[2,0].plot(cp_max_3_1[:,0], diff_3_1[cp_max_3_1[:,0],0], 'r^')
axes[2,0].plot(cp_max_3_1[:,1], diff_3_1[cp_max_3_1[:,1],1], 'g^')
axes[2,0].plot(cp_max_3_1[:,2], diff_3_1[cp_max_3_1[:,2],2], 'b^')
axes[2,0].plot(cp_max_begin_3_1, np.zeros_like(cp_max_begin_3_1), 'ko')

axes[2,1].plot(cp_max_3_2[:,0], diff_3_2[cp_max_3_2[:,0],0], 'r^')
axes[2,1].plot(cp_max_3_2[:,1], diff_3_2[cp_max_3_2[:,1],1], 'g^')
axes[2,1].plot(cp_max_3_2[:,2], diff_3_2[cp_max_3_2[:,2],2], 'b^')
axes[2,1].plot(cp_max_begin_3_2, np.zeros_like(cp_max_begin_3_2), 'ko')

axes[2,2].plot(cp_max_3_3[:,0], diff_3_3[cp_max_3_3[:,0],0], 'r^')
axes[2,2].plot(cp_max_3_3[:,1], diff_3_3[cp_max_3_3[:,1],1], 'g^')
axes[2,2].plot(cp_max_3_3[:,2], diff_3_3[cp_max_3_3[:,2],2], 'b^')
axes[2,2].plot(cp_max_begin_3_3, np.zeros_like(cp_max_begin_3_3), 'ko')

axes[3,0].plot(cp_max_4_1[:,0], diff_4_1[cp_max_4_1[:,0],0], 'r^')
axes[3,0].plot(cp_max_4_1[:,1], diff_4_1[cp_max_4_1[:,1],1], 'g^')
axes[3,0].plot(cp_max_4_1[:,2], diff_4_1[cp_max_4_1[:,2],2], 'b^')
axes[3,0].plot(cp_max_begin_4_1, np.zeros_like(cp_max_begin_4_1), 'ko')

axes[3,1].plot(cp_max_4_2[:,0], diff_4_2[cp_max_4_2[:,0],0], 'r^')
axes[3,1].plot(cp_max_4_2[:,1], diff_4_2[cp_max_4_2[:,1],1], 'g^')
axes[3,1].plot(cp_max_4_2[:,2], diff_4_2[cp_max_4_2[:,2],2], 'b^')
axes[3,1].plot(cp_max_begin_4_2, np.zeros_like(cp_max_begin_4_2), 'ko')

axes[3,2].plot(cp_max_4_3[:,0], diff_4_3[cp_max_4_3[:,0],0], 'r^')
axes[3,2].plot(cp_max_4_3[:,1], diff_4_3[cp_max_4_3[:,1],1], 'g^')
axes[3,2].plot(cp_max_4_3[:,2], diff_4_3[cp_max_4_3[:,2],2], 'b^')
axes[3,2].plot(cp_max_begin_4_3, np.zeros_like(cp_max_begin_4_3), 'ko')

axes[4,0].plot(cp_max_5_1[:,0], diff_5_1[cp_max_5_1[:,0],0], 'r^')
axes[4,0].plot(cp_max_5_1[:,1], diff_5_1[cp_max_5_1[:,1],1], 'g^')
axes[4,0].plot(cp_max_5_1[:,2], diff_5_1[cp_max_5_1[:,2],2], 'b^')
axes[4,0].plot(cp_max_begin_5_1, np.zeros_like(cp_max_begin_5_1), 'ko')

axes[4,1].plot(cp_max_5_2[:,0], diff_5_2[cp_max_5_2[:,0],0], 'r^')
axes[4,1].plot(cp_max_5_2[:,1], diff_5_2[cp_max_5_2[:,1],1], 'g^')
axes[4,1].plot(cp_max_5_2[:,2], diff_5_2[cp_max_5_2[:,2],2], 'b^')
axes[4,1].plot(cp_max_begin_5_2, np.zeros_like(cp_max_begin_5_2), 'ko')

axes[4,2].plot(cp_max_5_3[:,0], diff_5_3[cp_max_5_3[:,0],0], 'r^')
axes[4,2].plot(cp_max_5_3[:,1], diff_5_3[cp_max_5_3[:,1],1], 'g^')
axes[4,2].plot(cp_max_5_3[:,2], diff_5_3[cp_max_5_3[:,2],2], 'b^')
axes[4,2].plot(cp_max_begin_5_3, np.zeros_like(cp_max_begin_5_3), 'ko')

axes[5,0].plot(cp_max_6_1[:,0], diff_6_1[cp_max_6_1[:,0],0], 'r^')
axes[5,0].plot(cp_max_6_1[:,1], diff_6_1[cp_max_6_1[:,1],1], 'g^')
axes[5,0].plot(cp_max_6_1[:,2], diff_6_1[cp_max_6_1[:,2],2], 'b^')
axes[5,0].plot(cp_max_begin_6_1, np.zeros_like(cp_max_begin_6_1), 'ko')

axes[5,1].plot(cp_max_6_2[:,0], diff_6_2[cp_max_6_2[:,0],0], 'r^')
axes[5,1].plot(cp_max_6_2[:,1], diff_6_2[cp_max_6_2[:,1],1], 'g^')
axes[5,1].plot(cp_max_6_2[:,2], diff_6_2[cp_max_6_2[:,2],2], 'b^')
axes[5,1].plot(cp_max_begin_6_2, np.zeros_like(cp_max_begin_6_2), 'ko')

axes[5,2].plot(cp_max_6_3[:,0], diff_6_3[cp_max_6_3[:,0],0], 'r^')
axes[5,2].plot(cp_max_6_3[:,1], diff_6_3[cp_max_6_3[:,1],1], 'g^')
axes[5,2].plot(cp_max_6_3[:,2], diff_6_3[cp_max_6_3[:,2],2], 'b^')
axes[5,2].plot(cp_max_begin_6_3, np.zeros_like(cp_max_begin_6_3), 'ko')















plt.show()
