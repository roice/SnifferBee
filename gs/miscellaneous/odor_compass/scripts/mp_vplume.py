import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d import proj3d # for annotation in 3D

fd = h5py.File('FOC_Record.h5', 'r+')
pos_mb = np.asarray([-0.591570, -1.203342, 1.401007])

f = open('mp.txt', 'w')
for i in range(200):
    pos_vp = fd['/Debug/particle_'+str(i)+'_plume_pos'][...]
    f.write('draw ')
    for j in range(len(pos_vp)-1):
        p = pos_vp[j] - pos_mb
        f.write('rp(('+str('%.4f'%p[0])+'u,'+str('%.4f'%p[1])+'u,'+str('%.4f'%p[2])+'u)) .. ')
    p = pos_vp[len(pos_vp)-1] - pos_mb
    f.write('rp(('+str('%.4f'%p[0])+'u,'+str('%.4f'%p[1])+'u,'+str('%.4f'%p[2])+'u));\n')
f.close()


fig = plt.figure(figsize=(16, 12), facecolor='w')
ax = fig.gca(projection='3d')
ax.view_init(elev=30, azim=-90)

for i in range(100):
    pos_vp = fd['/Debug/particle_'+str(i)+'_plume_pos'][...]
    ax.plot(pos_vp[:,0], pos_vp[:,1], pos_vp[:,2], lw=1.1)

ax.scatter(pos_mb[0], pos_mb[1], pos_mb[2], s= 10)

ax.set_aspect('equal')

plt.show()
