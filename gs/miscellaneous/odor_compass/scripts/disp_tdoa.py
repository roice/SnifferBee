import h5py
import numpy as np
import matplotlib.pyplot as plt

fd = h5py.File('FOC_Record.h5', 'r+')
tdoa_1 = fd['/FOC/mox_toa_1'][...]
abs_1 = fd['/FOC/mox_abs_1'][...]
abs_2 = fd['/FOC/mox_abs_2'][...]

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)

'''
ax.plot(abs_1[:,0], color='r')
ax.plot(abs_1[:,1], color='g')
ax.plot(abs_1[:,2], color='b')
'''
ax.plot(abs_2[:,0], color='r')
ax.plot(abs_2[:,1], color='g')
ax.plot(abs_2[:,2], color='b')

plt.show()
