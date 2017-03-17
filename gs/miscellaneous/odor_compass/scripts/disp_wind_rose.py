import h5py
import numpy as np
import math
import matplotlib.pyplot as plt

MOX_DAQ_FREQ = 20
START_UP_TIME = 15

fd = h5py.File('FOC_Record.h5', 'r+')
wind = fd['/FOC/wind'][...]

wind = wind[START_UP_TIME*MOX_DAQ_FREQ:]

theta = []
r = []
for i in range(len(wind)):
    theta.append(math.atan2(-wind[i,0], wind[i,1]))
    r.append(np.sqrt(wind[i,0]*wind[i,0]+wind[i,1]*wind[i,1]))

sum_x = 0
sum_y = 0
for i in range(len(wind)):
    sum_x += wind[i,0]
    sum_y += wind[i,1]
main_theta = math.atan2(-sum_x, sum_y)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='polar')

ax.scatter(theta, r, s=1, cmap=plt.cm.hsv)
ax.set_alpha(0.75)

#ax.scatter(main_theta, 10000, s=100)

plt.show()
