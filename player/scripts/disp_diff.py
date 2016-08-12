import h5py
import numpy as np
import matplotlib.pyplot as plt

fd = h5py.File('FOC_Record.h5', 'r+')
diff = fd['/FOC/mox_diff'][...]

diff = diff[2000:6000,:]

diff_f = diff[:,0]/np.std(diff[:,0])
diff_l = diff[:,1]/np.std(diff[:,1])
diff_r = diff[:,2]/np.std(diff[:,2])

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)

ax.plot(diff_f, color='r')
ax.plot(diff_l, color='y')
ax.plot(diff_r, color='b')

plt.show()
