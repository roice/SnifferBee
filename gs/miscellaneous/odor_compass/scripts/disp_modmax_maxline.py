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

maxline_levels = fd['/FOC/wt_maxline_levels_s2'][...]
maxline_t = fd['/FOC/wt_maxline_t_s2'][...]
maxline_value = fd['/FOC/wt_maxline_value_s2'][...]
for i in range(len(maxline_levels)):
    if maxline_t[i, 0] == 964:
        print "levels = " + str(maxline_levels[i])
        print "t = " + str(maxline_t[i])
        print "value = " + str(maxline_value[i])

fig, axes = plt.subplots(nrows=3, figsize=(8,6))

for i in range(NUM_SENSORS):
    for j in range(WT_LEVELS):
        ds_name = '/FOC/wt_maxima_s'+str(i)+'_l'+str(j)
        modmax = fd[ds_name][...]
        axes[i].scatter((modmax[:,0]+LEN_WAVELET-LEN_WAVELET/2-SIGNAL_DELAY*MOX_DAQ_FREQ*MOX_INTERP_FACTOR/2)/(float)(MOX_DAQ_FREQ*MOX_INTERP_FACTOR), modmax[:,1])

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
            axes[i].plot((maxline_t[k, 0:maxline_levels[k]]+LEN_WAVELET-LEN_WAVELET/2-SIGNAL_DELAY*MOX_DAQ_FREQ*MOX_INTERP_FACTOR/2)/(float)(MOX_DAQ_FREQ*MOX_INTERP_FACTOR), maxline_value[k, 0:maxline_levels[k]])
    axes[i].set_xlabel('time (s)')

plt.show()
