import h5py
import numpy as np
import matplotlib.pyplot as plt

fd = h5py.File('FOC_Record.h5', 'r+')
direction = fd['/FOC/direction'][...]

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)

ax.plot(direction)

plt.show()
