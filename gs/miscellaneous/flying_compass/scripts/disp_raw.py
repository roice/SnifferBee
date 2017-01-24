import h5py
import numpy as np
import matplotlib.pyplot as plt

MOX_DAQ_FREQ = 20

fd = h5py.File('FOC_Record.h5', 'r+')
s_readings = fd['/FOC/mox_reading']
pos = fd['FOC/position']
att = fd['FOC/attitude']

fig = plt.figure(figsize=(8,4))

ax = fig.add_subplot(111)

ax.plot(np.asarray(range(len(s_readings[:,0])))/(float)(MOX_DAQ_FREQ), s_readings[:,0], color='red', linestyle='-', linewidth=1.5, label='gas sensor 1')
ax.plot(np.asarray(range(len(s_readings[:,0])))/(float)(MOX_DAQ_FREQ), s_readings[:,1], color='green', linestyle='-', linewidth=1.5, label='gas sensor 2')
ax.plot(np.asarray(range(len(s_readings[:,0])))/(float)(MOX_DAQ_FREQ), s_readings[:,2], color='blue', linestyle='-', linewidth=1.5, label='gas sensor 3')

ax.set_xlabel('time (s)')
ax.set_ylabel('Voltage (V)')

plt.show()
