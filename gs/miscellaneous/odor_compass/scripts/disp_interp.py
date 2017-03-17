import h5py
import numpy as np
import matplotlib.pyplot as plt

MOX_DAQ_FREQ = 20
MOX_INTERP_FACTOR = 10
SIGNAL_DELAY = 2 # seconds

fd = h5py.File('FOC_Record.h5', 'r+')
interp_out = fd['/FOC/mox_interp'][...]

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)

ax.plot(np.asarray(range(len(interp_out[:,0])))/float(MOX_DAQ_FREQ*MOX_INTERP_FACTOR) - SIGNAL_DELAY/2, interp_out[:,0], color='red')
ax.plot(np.asarray(range(len(interp_out[:,0])))/float(MOX_DAQ_FREQ*MOX_INTERP_FACTOR) - SIGNAL_DELAY/2, interp_out[:,1], color='green')
ax.plot(np.asarray(range(len(interp_out[:,0])))/float(MOX_DAQ_FREQ*MOX_INTERP_FACTOR) - SIGNAL_DELAY/2, interp_out[:,2], color='blue')

ax.set_xlabel('time (s)')
ax.set_ylabel('Voltage (V)')

plt.show()
