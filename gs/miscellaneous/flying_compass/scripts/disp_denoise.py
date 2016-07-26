import h5py
import numpy as np
import matplotlib.pyplot as plt

fd = h5py.File('FOC_Record.h5', 'r+')
mox_reading = fd['/FOC/mox_reading'][...]
mox_ukf_out = fd['/FOC/mox_denoise'][...]

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)
ax.plot(mox_ukf_out)
#ax.plot(mox_reading)

plt.show()
