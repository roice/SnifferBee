import h5py
import numpy as np
import matplotlib.pyplot as plt

fd = h5py.File('FOC_Record.h5', 'r+')
mox_ukf_out = fd['mox_ukf_output'][...]

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)
ax.plot(mox_ukf_out)

plt.show()
