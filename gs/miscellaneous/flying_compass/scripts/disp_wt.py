import h5py
import numpy as np
import matplotlib.pyplot as plt

fd = h5py.File('FOC_Record.h5', 'r+')
wt_out = fd['/FOC/wt_out'][...]
wt_idx = fd['/FOC/wt_index'][...]
wvs = fd['/FOC/wvs'][...]
wvs_idx = fd['/FOC/wvs_index'][...]
modmax_s0 = fd['/FOC/wt_modmax_0'][...]
modmax_s1 = fd['/FOC/wt_modmax_1'][...]
modmax_s2 = fd['/FOC/wt_modmax_2'][...]
mox_interp = fd['/FOC/mox_interp']
modmax_num = fd['/FOC/wt_modmax_num'][...]

N = wt_idx[1]
M = wvs_idx[1]

'''
A = []
for i in range(len(wvs_idx)):
    A.append(wvs[wvs_idx[i]:wvs_idx[i]+M])
A = np.asarray(A)
B = []
for i in range(len(wt_idx)):
    B.append(wt_out[wt_idx[i]+400:wt_idx[i]+N-400,0])
B = np.asarray(B)
plt.figure(1)
plt.imshow(B, interpolation='nearest', cmap='gray')
plt.grid(True)
plt.axes().set_aspect('auto')
'''


fig, axes = plt.subplots(nrows=2, figsize=(6,6))

for i in range(modmax_num[0]):
    ds_name = '/FOC/wt_maxline_'+'0'+'_'+str(i)
    maxline = fd[ds_name][...]
    axes[1].plot(maxline[:,0], maxline[:,1], color='k')

axes[1].scatter(modmax_s0[:,0], modmax_s0[:,1])

#for i in range(len(wt_idx)):
#    axes[1].plot(wt_out[wt_idx[i]:wt_idx[i]+N, 0])
    #axes[1].plot(wt_out[wt_idx[i]:wt_idx[i]+N, 1])
    #axes[1].plot(wt_out[wt_idx[i]:wt_idx[i]+N, 2])

'''
for i in range(len(wvs_idx)):
    axes[0].plot(wvs[wvs_idx[i]:wvs_idx[i]+M])
'''
plt.show()
