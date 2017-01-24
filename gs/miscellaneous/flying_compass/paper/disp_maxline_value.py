import h5py
import numpy as np
import matplotlib.pyplot as plt

NUM_SENSORS = 3
MOX_DAQ_FREQ = 20
MOX_INTERP_FACTOR = 10
SIGNAL_DELAY = 2 # seconds
WT_LEVELS = 100
LEN_WAVELET = (3*MOX_DAQ_FREQ*MOX_INTERP_FACTOR)
LEN_RECENT_INFO = (15*MOX_DAQ_FREQ*MOX_INTERP_FACTOR)

fd = h5py.File('FOC_Record.h5', 'r+')

fig, axes = plt.subplots(nrows=3, figsize=(8,5))

'''
for i in range(NUM_SENSORS):
    for j in range(WT_LEVELS):
        ds_name = '/FOC/wt_maxima_s'+str(i)+'_l'+str(j)
        modmax = fd[ds_name][...]
        idx_begin = 0
        for k in range(len(modmax)):
            if modmax[k,0] < MOX_DAQ_FREQ*MOX_INTERP_FACTOR*20:
                continue
            else:
                idx_begin = k
                break
        idx_end = idx_begin
        for k in range(len(modmax)):
            if modmax[k,0] < MOX_DAQ_FREQ*MOX_INTERP_FACTOR*35:
                idx_end = k
                continue
            else:
                break
        axes[i].scatter((modmax[idx_begin:idx_end,0]+LEN_WAVELET/2-MOX_DAQ_FREQ*MOX_INTERP_FACTOR*20)/(float)(MOX_DAQ_FREQ*MOX_INTERP_FACTOR), modmax[idx_begin:idx_end,1], s=10)
'''

for i in range(NUM_SENSORS):
    for j in range(2):
        if j == 0:
            ds_name_levels = '/FOC/wt_maxline_levels_s'+str(i)
            ds_name_t = '/FOC/wt_maxline_t_s'+str(i)
            ds_name_value = '/FOC/wt_maxline_value_s'+str(i)
        else:
            ds_name_levels = '/FOC/wt_minline_levels_s'+str(i)
            ds_name_t = '/FOC/wt_minline_t_s'+str(i)
            ds_name_value = '/FOC/wt_minline_value_s'+str(i)
        maxline_levels = fd[ds_name_levels][...]
        maxline_t = fd[ds_name_t][...]
        maxline_value = fd[ds_name_value][...]

        idx_begin = 0
        for k in range(len(maxline_t)):
            if maxline_t[k,0] < MOX_DAQ_FREQ*MOX_INTERP_FACTOR*20-LEN_WAVELET+LEN_WAVELET/2+SIGNAL_DELAY*MOX_DAQ_FREQ*MOX_INTERP_FACTOR/2:
                continue
            else:
                idx_begin = k
                break
        idx_end = idx_begin
        for k in range(len(maxline_t)):
            if maxline_t[k,0] < MOX_DAQ_FREQ*MOX_INTERP_FACTOR*35-LEN_WAVELET+LEN_WAVELET/2+SIGNAL_DELAY*MOX_DAQ_FREQ*MOX_INTERP_FACTOR/2:
                idx_end = k
                continue
            else:
                break

        for k in range(len(maxline_levels)):
            if k < idx_begin or k > idx_end:
                continue
            axes[i].plot((maxline_t[k, 0:maxline_levels[k]]-(MOX_DAQ_FREQ*MOX_INTERP_FACTOR*20-LEN_WAVELET+LEN_WAVELET/2+SIGNAL_DELAY*MOX_DAQ_FREQ*MOX_INTERP_FACTOR/2))/(float)(MOX_DAQ_FREQ*MOX_INTERP_FACTOR), maxline_value[k, 0:maxline_levels[k]], linewidth=1, color='k')
    axes[i].set_xlabel('time (s)')
    axes[i].set_ylabel('WT of sensor '+str(i+1))

plt.savefig('maxline_value.pdf', format='pdf', bbox_inches="tight")

plt.show()
