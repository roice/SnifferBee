import h5py
import numpy as np
import matplotlib.pyplot as plt

MOX_DAQ_FREQ = 20
MOX_INTERP_FACTOR = 10
WT_LEVELS = 100
LEN_WAVELET = (3*MOX_DAQ_FREQ*MOX_INTERP_FACTOR)
LEN_RECENT_INFO = (15*MOX_DAQ_FREQ*MOX_INTERP_FACTOR)

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
maxline_num = [modmax_num[0*WT_LEVELS], modmax_num[1*WT_LEVELS], modmax_num[2*WT_LEVELS]]
s_readings = fd['/FOC/mox_reading'][...]
interp_out = fd['/FOC/mox_interp'][...]

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

fig, axes = plt.subplots(nrows=4, figsize=(6,6))

axes[0].plot(s_readings[:,0], color='red')
axes[0].plot(s_readings[:,1], color='green')
axes[0].plot(s_readings[:,2], color='blue')

axes[1].plot(interp_out)

for i in range(len(wt_idx)):
    axes[2].plot(wt_out[wt_idx[i]+LEN_WAVELET/2:wt_idx[i]+LEN_WAVELET/2+LEN_RECENT_INFO-1, 0])
    #axes[2].plot(wt_out[wt_idx[i]:wt_idx[i]+N, 1])
    #axes[2].plot(wt_out[wt_idx[i]:wt_idx[i]+N, 2])

axes[2].scatter(modmax_s0[:,0]-LEN_WAVELET/2, modmax_s0[:,1])

for i in range(maxline_num[0]):
    ds_name = '/FOC/wt_maxline_'+'0'+'_'+str(i)
    maxline = fd[ds_name][...]
    axes[3].plot(maxline[:,0] -LEN_WAVELET/2, maxline[:,2], color='r')
for i in range(maxline_num[1]):
    ds_name = '/FOC/wt_maxline_'+'1'+'_'+str(i)
    maxline = fd[ds_name][...]
    axes[3].plot(maxline[:,0] -LEN_WAVELET/2, maxline[:,2], color='g')
for i in range(maxline_num[2]):
    ds_name = '/FOC/wt_maxline_'+'2'+'_'+str(i)
    maxline = fd[ds_name][...]
    axes[3].plot(maxline[:,0] -LEN_WAVELET/2, maxline[:,2], color='b')





'''
for i in range(len(wvs_idx)):
    axes[0].plot(wvs[wvs_idx[i]:wvs_idx[i]+M])
'''
plt.show()
