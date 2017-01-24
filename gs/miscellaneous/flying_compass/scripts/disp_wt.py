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
wt_out = fd['/FOC/wt_out'][...]
wvs = fd['/FOC/wvs'][...]
wvs_idx = fd['/FOC/wvs_index'][...]

fig, axes = plt.subplots(nrows=NUM_SENSORS, figsize=(8,5))

for i in range(NUM_SENSORS):
    for j in range(WT_LEVELS):
        axes[i].plot((np.asarray(range(len(wt_out[:,j,i])))+LEN_WAVELET-LEN_WAVELET/2-SIGNAL_DELAY*MOX_DAQ_FREQ*MOX_INTERP_FACTOR/2)/(float)(MOX_DAQ_FREQ*MOX_INTERP_FACTOR), wt_out[:, j, i])
    axes[i].set_xlabel('time (s)')

plt.show()
