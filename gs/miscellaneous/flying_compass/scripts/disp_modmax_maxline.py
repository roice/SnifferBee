import h5py
import numpy as np
import matplotlib.pyplot as plt

NUM_SENSORS = 3
MOX_DAQ_FREQ = 20
MOX_INTERP_FACTOR = 10
WT_LEVELS = 100
LEN_WAVELET = (3*MOX_DAQ_FREQ*MOX_INTERP_FACTOR)
LEN_RECENT_INFO = (15*MOX_DAQ_FREQ*MOX_INTERP_FACTOR)

fd = h5py.File('FOC_Record.h5', 'r+')

fig, axes = plt.subplots(nrows=3, figsize=(8,6))

for i in range(NUM_SENSORS):
    for j in range(WT_LEVELS):
        ds_name = '/FOC/wt_maxima_s'+str(i)+'_l'+str(j)
        modmax = fd[ds_name][...]
        axes[i].scatter(modmax[:,0], modmax[:,1])

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
        for k in range(len(maxline_levels)):
            axes[i].plot(maxline_t[k, 0:maxline_levels[k]], maxline_value[k, 0:maxline_levels[k]])

plt.show()
