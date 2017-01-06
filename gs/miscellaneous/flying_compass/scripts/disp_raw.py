import h5py
import numpy as np
import matplotlib.pyplot as plt

MOX_DAQ_FREQ = 20
MOX_DAQ_INTERP_FACTOR = 10

fd = h5py.File('FOC_Record.h5', 'r+')
s_readings = fd['/FOC/mox_reading']
pos = fd['FOC/position']
att = fd['FOC/attitude']

fig = plt.figure(figsize=(6,6))

ax = fig.add_subplot(111)

ax.plot(np.asarray(range(len(s_readings[:,0])))/(float)(MOX_DAQ_FREQ), s_readings[:,0], color='red')
ax.plot(np.asarray(range(len(s_readings[:,0])))/(float)(MOX_DAQ_FREQ), s_readings[:,1], color='green')
ax.plot(np.asarray(range(len(s_readings[:,0])))/(float)(MOX_DAQ_FREQ), s_readings[:,2], color='blue')

ax.set_xlabel('time (s)')
ax.set_ylabel('Voltage (V)')

plt.show()
