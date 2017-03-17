import h5py
import numpy as np
import matplotlib.pyplot as plt

fd = h5py.File('FOC_Record.h5', 'r+')
mox_reading = fd['/FOC/mox_reading'][...]
mox_ukf_out = fd['/FOC/mox_denoise'][...]

fig, (ax1, ax2) = plt.subplots(nrows=2)

ax1.plot(mox_reading[:,0], color='red')
ax1.plot(mox_reading[:,1], color='green')
ax1.plot(mox_reading[:,2], color='blue')

ax1.plot(mox_ukf_out[:,0], color='red')
ax1.plot(mox_ukf_out[:,1], color='green')
ax1.plot(mox_ukf_out[:,2], color='blue')

ax2.plot(mox_ukf_out[:,0], color='red')
ax2.plot(mox_ukf_out[:,1], color='green')
ax2.plot(mox_ukf_out[:,2], color='blue')

plt.show()
