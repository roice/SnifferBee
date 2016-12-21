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
wt_out = fd['/FOC/wt_out'][...]
wvs = fd['/FOC/wvs'][...]
wvs_idx = fd['/FOC/wvs_index'][...]

fig, axes = plt.subplots(nrows=NUM_SENSORS, figsize=(6,6))

for i in range(NUM_SENSORS):
    for j in range(WT_LEVELS):
        axes[i].plot(wt_out[:, j, i])

plt.show()
