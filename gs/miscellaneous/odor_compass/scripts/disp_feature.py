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

feature_idx = fd['FOC/feature_idx_ml'][...]

print feature_idx

fig, axes = plt.subplots(nrows=3, figsize=(8,6))

plt.show()
