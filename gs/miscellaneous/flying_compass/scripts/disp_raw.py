import h5py
import numpy as np
import matplotlib.pyplot as plt

#fd = h5py.File('../data/Record_2016-07-08_15-24-37.h5', 'r+')
#fd = h5py.File('../data/Record_2016-07-08_15-27-30.h5', 'r+')
#fd = h5py.File('../data/Record_2016-07-08_15-30-28.h5', 'r+')
#fd = h5py.File('../data/Record_2016-07-08_15-32-12.h5', 'r+')
#fd = h5py.File('../data/Record_2016-07-08_15-35-47.h5', 'r+')
#fd = h5py.File('../data/Record_2016-07-08_15-38-16.h5', 'r+')
#fd = h5py.File('../data/Record_2016-08-03_17-30-06.h5', 'r+')
#fd = h5py.File('../data/Record_2016-08-19_16-36-49.h5', 'r+')
#fd = h5py.File('../data/Record_2016-08-19_16-45-13.h5', 'r+')
#s_readings = fd['robot1/mox'][...]
fd = h5py.File('FOC_Record.h5', 'r+')
s_readings = fd['/FOC/mox_reading']

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)
ax.plot(s_readings[:,0], color='red')
ax.plot(s_readings[:,1], color='green')
ax.plot(s_readings[:,2], color='blue')

plt.show()
