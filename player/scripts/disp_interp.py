import h5py
import numpy as np
import matplotlib.pyplot as plt

fd = h5py.File('FOC_Record.h5', 'r+')
interp_out = fd['/FOC/mox_interp'][...]

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)
ax.plot(interp_out)

plt.show()
