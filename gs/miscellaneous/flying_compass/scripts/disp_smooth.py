import h5py
import numpy as np
import matplotlib.pyplot as plt

fd = h5py.File('FOC_Record.h5', 'r+')
sm_1_1 = fd['/FOC/mox_smooth_group_1_layer_1'][750:]
sm_1_2 = fd['/FOC/mox_smooth_group_1_layer_2'][750:]
sm_1_3 = fd['/FOC/mox_smooth_group_1_layer_3'][750:]
sm_1_4 = fd['/FOC/mox_smooth_group_1_layer_4'][750:]
sm_2_1 = fd['/FOC/mox_smooth_group_2_layer_1'][750:]
sm_2_2 = fd['/FOC/mox_smooth_group_2_layer_2'][750:]
sm_2_3 = fd['/FOC/mox_smooth_group_2_layer_3'][750:]
sm_2_4 = fd['/FOC/mox_smooth_group_2_layer_4'][750:]
sm_3_1 = fd['/FOC/mox_smooth_group_3_layer_1'][750:]
sm_3_2 = fd['/FOC/mox_smooth_group_3_layer_2'][750:]
sm_3_3 = fd['/FOC/mox_smooth_group_3_layer_3'][750:]
sm_3_4 = fd['/FOC/mox_smooth_group_3_layer_4'][750:]
sm_4_1 = fd['/FOC/mox_smooth_group_4_layer_1'][750:]
sm_4_2 = fd['/FOC/mox_smooth_group_4_layer_2'][750:]
sm_4_3 = fd['/FOC/mox_smooth_group_4_layer_3'][750:]
sm_4_4 = fd['/FOC/mox_smooth_group_4_layer_4'][750:]
sm_5_1 = fd['/FOC/mox_smooth_group_5_layer_1'][750:]
sm_5_2 = fd['/FOC/mox_smooth_group_5_layer_2'][750:]
sm_5_3 = fd['/FOC/mox_smooth_group_5_layer_3'][750:]
sm_5_4 = fd['/FOC/mox_smooth_group_5_layer_4'][750:]
sm_6_1 = fd['/FOC/mox_smooth_group_6_layer_1'][750:]
sm_6_2 = fd['/FOC/mox_smooth_group_6_layer_2'][750:]
sm_6_3 = fd['/FOC/mox_smooth_group_6_layer_3'][750:]
sm_6_4 = fd['/FOC/mox_smooth_group_6_layer_4'][750:]

fig, axes = plt.subplots(nrows=6, ncols=4, figsize=(6, 6))

axes[0,0].plot(sm_1_1[:,0], color='red')
axes[0,0].plot(sm_1_1[:,1], color='green')
axes[0,0].plot(sm_1_1[:,2], color='blue')

axes[0,1].plot(sm_1_2[:,0], color='red')
axes[0,1].plot(sm_1_2[:,1], color='green')
axes[0,1].plot(sm_1_2[:,2], color='blue')

axes[0,2].plot(sm_1_3[:,0], color='red')
axes[0,2].plot(sm_1_3[:,1], color='green')
axes[0,2].plot(sm_1_3[:,2], color='blue')

axes[0,3].plot(sm_1_4[:,0], color='red')
axes[0,3].plot(sm_1_4[:,1], color='green')
axes[0,3].plot(sm_1_4[:,2], color='blue')

axes[1,0].plot(sm_2_1[:,0], color='red')
axes[1,0].plot(sm_2_1[:,1], color='green')
axes[1,0].plot(sm_2_1[:,2], color='blue')

axes[1,1].plot(sm_2_2[:,0], color='red')
axes[1,1].plot(sm_2_2[:,1], color='green')
axes[1,1].plot(sm_2_2[:,2], color='blue')

axes[1,2].plot(sm_2_3[:,0], color='red')
axes[1,2].plot(sm_2_3[:,1], color='green')
axes[1,2].plot(sm_2_3[:,2], color='blue')

axes[1,3].plot(sm_2_4[:,0], color='red')
axes[1,3].plot(sm_2_4[:,1], color='green')
axes[1,3].plot(sm_2_4[:,2], color='blue')

axes[2,0].plot(sm_3_1[:,0], color='red')
axes[2,0].plot(sm_3_1[:,1], color='green')
axes[2,0].plot(sm_3_1[:,2], color='blue')

axes[2,1].plot(sm_3_2[:,0], color='red')
axes[2,1].plot(sm_3_2[:,1], color='green')
axes[2,1].plot(sm_3_2[:,2], color='blue')

axes[2,2].plot(sm_3_3[:,0], color='red')
axes[2,2].plot(sm_3_3[:,1], color='green')
axes[2,2].plot(sm_3_3[:,2], color='blue')

axes[2,3].plot(sm_3_4[:,0], color='red')
axes[2,3].plot(sm_3_4[:,1], color='green')
axes[2,3].plot(sm_3_4[:,2], color='blue')

axes[3,0].plot(sm_4_1[:,0], color='red')
axes[3,0].plot(sm_4_1[:,1], color='green')
axes[3,0].plot(sm_4_1[:,2], color='blue')

axes[3,1].plot(sm_4_2[:,0], color='red')
axes[3,1].plot(sm_4_2[:,1], color='green')
axes[3,1].plot(sm_4_2[:,2], color='blue')

axes[3,2].plot(sm_4_3[:,0], color='red')
axes[3,2].plot(sm_4_3[:,1], color='green')
axes[3,2].plot(sm_4_3[:,2], color='blue')

axes[3,3].plot(sm_4_4[:,0], color='red')
axes[3,3].plot(sm_4_4[:,1], color='green')
axes[3,3].plot(sm_4_4[:,2], color='blue')

axes[4,0].plot(sm_5_1[:,0], color='red')
axes[4,0].plot(sm_5_1[:,1], color='green')
axes[4,0].plot(sm_5_1[:,2], color='blue')

axes[4,1].plot(sm_5_2[:,0], color='red')
axes[4,1].plot(sm_5_2[:,1], color='green')
axes[4,1].plot(sm_5_2[:,2], color='blue')

axes[4,2].plot(sm_5_3[:,0], color='red')
axes[4,2].plot(sm_5_3[:,1], color='green')
axes[4,2].plot(sm_5_3[:,2], color='blue')

axes[4,3].plot(sm_5_4[:,0], color='red')
axes[4,3].plot(sm_5_4[:,1], color='green')
axes[4,3].plot(sm_5_4[:,2], color='blue')

axes[5,0].plot(sm_6_1[:,0], color='red')
axes[5,0].plot(sm_6_1[:,1], color='green')
axes[5,0].plot(sm_6_1[:,2], color='blue')

axes[5,1].plot(sm_6_2[:,0], color='red')
axes[5,1].plot(sm_6_2[:,1], color='green')
axes[5,1].plot(sm_6_2[:,2], color='blue')

axes[5,2].plot(sm_6_3[:,0], color='red')
axes[5,2].plot(sm_6_3[:,1], color='green')
axes[5,2].plot(sm_6_3[:,2], color='blue')

axes[5,3].plot(sm_6_4[:,0], color='red')
axes[5,3].plot(sm_6_4[:,1], color='green')
axes[5,3].plot(sm_6_4[:,2], color='blue')




















plt.show()
