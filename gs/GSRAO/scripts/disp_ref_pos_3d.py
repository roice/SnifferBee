import h5py
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fd = h5py.File('./Record_2017-06-28_12-14-56.h5', 'r+')

ref_pos = fd['robot1/ref_enu'][...]

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.scatter(ref_pos[:,0], ref_pos[:,1], ref_pos[:,2])

ax.set_xlabel('West -- East')
ax.set_ylabel('South -- North')

ax.axis('equal')
plt.show()
