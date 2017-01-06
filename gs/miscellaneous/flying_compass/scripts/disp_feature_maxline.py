import h5py
import numpy as np
import matplotlib.pyplot as plt
import random

NUM_SENSORS = 3
MOX_DAQ_FREQ = 20
MOX_INTERP_FACTOR = 10
WT_LEVELS = 100
LEN_WAVELET = (3*MOX_DAQ_FREQ*MOX_INTERP_FACTOR)
LEN_RECENT_INFO = (15*MOX_DAQ_FREQ*MOX_INTERP_FACTOR)

fd = h5py.File('FOC_Record.h5', 'r+')

feature_type = fd['/FOC/feature_type'][...]
feature_idx_ml = fd['/FOC/feature_idx_ml'][...]

color_map = np.zeros((len(feature_type), 3))
for i in range(len(feature_type)):
    for j in range(3):
        color_map[i,j] = random.random()

fig, axes = plt.subplots(nrows=3, figsize=(8,6))

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
        axes[idx].plot((maxline_t[idx_ml, 0:maxline_levels[idx_ml]]+LEN_WAVELET/2)/(float)(MOX_DAQ_FREQ*MOX_INTERP_FACTOR), maxline_value[idx_ml, 0:maxline_levels[idx_ml]], color = (color_map[i,0], color_map[i,1], color_map[i,2]))

for i in range(3):
    axes[i].set_xlabel('time (s)')

plt.show()
