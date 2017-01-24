import h5py
import numpy as np
import matplotlib.pyplot as plt
import random

NUM_SENSORS = 3
MOX_DAQ_FREQ = 20
MOX_INTERP_FACTOR = 10
SIGNAL_DELAY = 2 # seconds
WT_LEVELS = 100
LEN_WAVELET = (3*MOX_DAQ_FREQ*MOX_INTERP_FACTOR)
LEN_RECENT_INFO = (15*MOX_DAQ_FREQ*MOX_INTERP_FACTOR)

fd = h5py.File('FOC_Record.h5', 'r+')

feature_type = fd['/FOC/feature_type'][...]
feature_idx_ml = fd['/FOC/feature_idx_ml'][...]

fig, axes = plt.subplots(nrows=3, figsize=(8,5))

color_map = ['b','g','r','c','m','y','k']


for idx in range(NUM_SENSORS):
    max_ds_name_levels = '/FOC/wt_maxline_levels_s'+str(idx)
    max_ds_name_t = '/FOC/wt_maxline_t_s'+str(idx)
    max_ds_name_value = '/FOC/wt_maxline_value_s'+str(idx)
    min_ds_name_levels = '/FOC/wt_minline_levels_s'+str(idx)
    min_ds_name_t = '/FOC/wt_minline_t_s'+str(idx)
    min_ds_name_value = '/FOC/wt_minline_value_s'+str(idx)
    for i in range(len(feature_type)):
        if feature_type[i] == 0:
            maxline_levels = fd[max_ds_name_levels][...]
            maxline_t = fd[max_ds_name_t][...]
            maxline_value = fd[max_ds_name_value][...]
        else:
            maxline_levels = fd[min_ds_name_levels][...]
            maxline_t = fd[min_ds_name_t][...]
            maxline_value = fd[min_ds_name_value][...]
        idx_ml = feature_idx_ml[i,idx]

        if maxline_t[idx_ml, 0] < MOX_DAQ_FREQ*MOX_INTERP_FACTOR*20-LEN_WAVELET+LEN_WAVELET/2+SIGNAL_DELAY*MOX_DAQ_FREQ*MOX_INTERP_FACTOR/2:
            continue
        elif maxline_t[idx_ml, 0] > MOX_DAQ_FREQ*MOX_INTERP_FACTOR*35-LEN_WAVELET+LEN_WAVELET/2+SIGNAL_DELAY*MOX_DAQ_FREQ*MOX_INTERP_FACTOR/2:
            continue

        axes[idx].plot((maxline_t[idx_ml, 0:maxline_levels[idx_ml]]-(MOX_DAQ_FREQ*MOX_INTERP_FACTOR*20-LEN_WAVELET+LEN_WAVELET/2+SIGNAL_DELAY*MOX_DAQ_FREQ*MOX_INTERP_FACTOR/2))/(float)(MOX_DAQ_FREQ*MOX_INTERP_FACTOR), maxline_value[idx_ml, 0:maxline_levels[idx_ml]], color = color_map[i%len(color_map)])

for i in range(3):
    axes[i].set_xlabel('time (s)')
    axes[i].set_ylabel('WT of sensor '+str(i+1))

plt.savefig('grouping_result.pdf', format='pdf', bbox_inches="tight")

plt.show()
