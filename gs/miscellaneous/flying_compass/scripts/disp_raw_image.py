import h5py
import numpy as np
import matplotlib.pyplot as plt

fd = h5py.File('../data/Record_2016-08-03_17-30-06.h5', 'r+')

s_readings = fd['robot1/mox'][...]

'''
m = np.arange(len(s_readings[:,0])*len(s_readings[:,0])).reshape(len(s_readings[:,0]), len(s_readings[:,0]))
for i in range(len(s_readings[:,0])):
    for j in range(len(s_readings[:,0])):
        m[i,j] = (s_readings[i,0]+s_readings[i,1])/6.6
'''

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)
'''
ax.plot(s_readings[:,0], color='red')
ax.plot(s_readings[:,1], color='yellow')
ax.plot(s_readings[:,2], color='blue')
'''
ax.imshow(m)

plt.show()
