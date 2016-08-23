import h5py
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter

#fd = h5py.File('../data/Record_2016-08-03_17-30-06.h5', 'r+')
#fd = h5py.File('../data/Record_2016-08-19_16-45-13.h5', 'r+')
#fd = h5py.File('../data/Record_2016-08-19_16-52-27.h5', 'r+')
fd = h5py.File('FOC_Record.h5', 'r+')
wind = fd['/FOC/wind'][...]

'''
wind_f_x = gaussian_filter(wind[:,0], sigma = 50)
wind_f_y = gaussian_filter(wind[:,1], sigma = 50)
wind_f_z = gaussian_filter(wind[:,2], sigma = 50)
'''

wind_f_x = wind[:,0]
wind_f_y = wind[:,1]
wind_f_z = wind[:,2]

direction = []
for i in range(len(wind)):
    direction.append(math.atan2(-wind_f_x[i], wind_f_y[i]))
direction = np.asarray(direction)

strength = []
for i in range(len(wind)):
    strength.append(wind_f_x[i]*wind_f_x[i] + wind_f_y[i]*wind_f_y[i])
strength = np.asarray(strength)

w_d = []
n = 100
for i in range(len(direction)):
    if (i < n):
        continue
    sum_w = 0
    sum_w_d = 0
    for j in range(n):
        sum_w += strength[i-j]
        sum_w_d += direction[i-j]*strength[i-j]
    w_d.append(sum_w_d/sum_w)
w_d = np.asarray(w_d)

v= []
n = 25*60
for i in range(len(wind)):
    if (i < n):
        continue
    sum_x = 0
    sum_y = 0
    for j in range(n):
        sum_x += wind[i,0]
        sum_y += wind[i,1]
    v.append(math.atan2(-sum_x, sum_y))
v = np.asarray(v)

theta = []
r = []
for i in range(len(wind)):
    theta.append(math.atan2(-wind[i,0], wind[i,1]))
    r.append(wind[i,0]*wind[i,0]+wind[i,1]*wind[i,1])

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

ax.scatter(main_theta, 10000, s=100)

plt.show()
