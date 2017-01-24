import h5py
import numpy as np
import math
import matplotlib.pyplot as plt

fd = h5py.File('FOC_Record.h5', 'r+')
wind = fd['/FOC/wind'][...]
att = fd['/FOC/attitude'][...]

theta = []
r = []
for i in range(len(wind)):
    theta.append(math.atan2(-wind[i,0], wind[i,1]))
    r.append(wind[i,0]*wind[i,0]+wind[i,1]*wind[i,1])
theta = np.asarray(theta)

sum_x = 0
sum_y = 0
for i in range(len(wind)):
    sum_x += wind[i,0]
    sum_y += wind[i,1]
main_theta = math.atan2(-sum_x, sum_y)

fig, (ax1, ax2) = plt.subplots(nrows=2)

ax1.plot(theta*180./np.pi, color='b')
ax1.plot(att[:,2]*180./np.pi, color='k')
ax1.plot((theta+att[:,2])*180./np.pi, color='r')

ax2.plot(r)

plt.show()
