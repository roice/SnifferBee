import h5py
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter

fd = h5py.File('FOC_Record.h5', 'r+')
mox_reading = fd['/FOC/mox_reading'][...]

f_1 = gaussian_filter(mox_reading[:,0], sigma=5)
l_1 = gaussian_filter(mox_reading[:,1], sigma=5)
r_1 = gaussian_filter(mox_reading[:,2], sigma=5)
f_2 = gaussian_filter(mox_reading[:,0], sigma=1.1892*5)
l_2 = gaussian_filter(mox_reading[:,1], sigma=1.1892*5)
r_2 = gaussian_filter(mox_reading[:,2], sigma=1.1892*5)
f = f_1 - f_2
l = l_1 - l_2
r = r_1 - r_2

fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4)

ax1.plot(mox_reading[:,0], color='r')
ax1.plot(mox_reading[:,1], color='g')
ax1.plot(mox_reading[:,2], color='b')

ax2.plot(f_1, color='r')
ax2.plot(l_1, color='g')
ax2.plot(r_1, color='b')

ax3.plot(f_2, color='r')
ax3.plot(l_2, color='g')
ax3.plot(r_2, color='b')

ax4.plot(f, color = 'r')
ax4.plot(l, color = 'g')
ax4.plot(r, color = 'b')

plt.show()
