import h5py
import numpy as np
import matplotlib.pyplot as plt

MOX_DAQ_FREQ = 20
MOX_DAQ_INTERP_FACTOR = 10

fd = h5py.File('FOC_Record.h5', 'r+')
s_readings = fd['/FOC/mox_reading']
pos = fd['FOC/position']
att = fd['FOC/attitude']

s_readings = s_readings[MOX_DAQ_FREQ*20:MOX_DAQ_FREQ*35]

fig = plt.figure(figsize=(8,4))

ax = fig.add_subplot(111)

ax.plot(np.asarray(range(len(s_readings[:,0])))/(float)(MOX_DAQ_FREQ), s_readings[:,0], color='red', linestyle='-', linewidth=1.5, label='gas sensor 1')
ax.plot(np.asarray(range(len(s_readings[:,0])))/(float)(MOX_DAQ_FREQ), s_readings[:,1], color='green', linestyle='-', linewidth=1.5, label='gas sensor 2')
ax.plot(np.asarray(range(len(s_readings[:,0])))/(float)(MOX_DAQ_FREQ), s_readings[:,2], color='blue', linestyle='-', linewidth=1.5, label='gas sensor 3')

ax.set_xlabel('time (s)')
ax.set_ylabel('Voltage (V)')

lgnd = ax.legend(loc='lower left', fontsize=11)
lgnd.legendHandles[0]._sizes = [30]
lgnd.legendHandles[1]._sizes = [30]

plt.savefig('raw_signals.pdf', format='pdf', bbox_inches="tight")

plt.show()
