import h5py
import numpy as np
import matplotlib.pyplot as plt

fd = h5py.File('./Record_2017-01-20_19-57-10.h5', 'r+')
s_readings = fd['robot1/mox']

fig,axes = plt.subplots(nrows=2,figsize=(6,6))

axes[0].plot(s_readings[:,0], color='red')
axes[0].plot(s_readings[:,1], color='green')
axes[0].plot(s_readings[:,2], color='blue')

plt.show()
