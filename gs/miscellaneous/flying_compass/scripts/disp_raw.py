import h5py
import numpy as np
import matplotlib.pyplot as plt

fd = h5py.File('../data/Record_2016-07-08_15-24-37.h5', 'r+')
s_readings = fd['sensors_of_robot_0'][...]

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)
ax.plot(s_readings)

plt.show()
