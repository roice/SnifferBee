import h5py
import numpy as np
import matplotlib.pyplot as plt

fd = h5py.File('FOC_Record.h5', 'r+')
wt_out = fd['/FOC/wt_out'][...]
wt_idx = fd['/FOC/wt_index'][...]
wvs = fd['/FOC/wvs'][...]
wvs_idx = fd['/FOC/wvs_index'][...]
mox_interp = fd['/FOC/mox_interp']
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
for i in range(len(wvs_idx)):
    axes[0].plot(wvs[wvs_idx[i]:wvs_idx[i]+M])
for i in range(len(wt_idx)):
    axes[1].plot(wt_out[wt_idx[i]:wt_idx[i]+N, 0])
    #axes[1].plot(wt_out[wt_idx[i]:wt_idx[i]+N, 1])
    #axes[1].plot(wt_out[wt_idx[i]:wt_idx[i]+N, 2])

plt.show()
