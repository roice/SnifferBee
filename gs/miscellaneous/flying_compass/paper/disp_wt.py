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
wt_out = fd['/FOC/wt_out'][...]

wt_out = wt_out[MOX_DAQ_FREQ*MOX_INTERP_FACTOR*20-LEN_WAVELET+LEN_WAVELET/2+SIGNAL_DELAY*MOX_DAQ_FREQ*MOX_INTERP_FACTOR/2:MOX_DAQ_FREQ*MOX_INTERP_FACTOR*35-LEN_WAVELET+LEN_WAVELET/2+SIGNAL_DELAY*MOX_DAQ_FREQ*MOX_INTERP_FACTOR/2]

fig, axes = plt.subplots(nrows=NUM_SENSORS, figsize=(8,5))

color_map = ['b','g','r','c','m','y','k']

for i in range(NUM_SENSORS):
    for j in range(WT_LEVELS):
        axes[i].plot((np.asarray(range(len(wt_out[:,j,i]))))/(float)(MOX_DAQ_FREQ*MOX_INTERP_FACTOR), wt_out[:, j, i], linewidth=0.4, color = color_map[j%len(color_map)])
    axes[i].set_xlabel('time (s)')
    axes[i].set_ylabel('WT of sensor '+str(i+1))

plt.savefig('wt_signals.pdf', format='pdf', bbox_inches="tight")

plt.show()
