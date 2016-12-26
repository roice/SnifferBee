import h5py
import numpy as np
import matplotlib.pyplot as plt

MOX_DAQ_FREQ = 20
MOX_DAQ_INTERP_FACTOR = 10

fd = h5py.File('FOC_Record.h5', 'r+')
s_readings = fd['/FOC/mox_reading']
pos = fd['FOC/position']
att = fd['FOC/attitude']

fig,axes = plt.subplots(nrows=2,figsize=(6,6))

axes[0].plot(np.asarray(range(len(s_readings[:,0])))/(float)(MOX_DAQ_FREQ), s_readings[:,0], color='red')
axes[0].plot(np.asarray(range(len(s_readings[:,0])))/(float)(MOX_DAQ_FREQ), s_readings[:,1], color='green')
axes[0].plot(np.asarray(range(len(s_readings[:,0])))/(float)(MOX_DAQ_FREQ), s_readings[:,2], color='blue')

axes[1].plot(att[:,0]*180./np.pi, color='red')
axes[1].plot(att[:,1]*180./np.pi, color='green')
axes[1].plot(att[:,2]*180./np.pi, color='blue')

plt.show()
