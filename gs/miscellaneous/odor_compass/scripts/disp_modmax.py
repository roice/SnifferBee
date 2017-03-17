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
        print "num of maxima of sensor " + str(i) + ' level ' + str(j) + ' is ' + str(len(modmax))
        axes[i].scatter((modmax[:,0]+LEN_WAVELET/2)/(float)(MOX_DAQ_FREQ*MOX_INTERP_FACTOR), modmax[:,1])
    axes[i].set_xlabel('time (s)')

plt.show()
