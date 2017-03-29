import h5py
import numpy as np
import matplotlib.pyplot as plt

fd = h5py.File('./Record_2017-03-28_22-40-31.h5', 'r+')

motors = fd['robot1/motors']
bat_volt = fd['robot1/bat_volt']

U0 = (pow(motors[:,0],2) + pow(motors[:,1],2) + pow(motors[:,2],2) + pow(motors[:,3],2))*bat_volt[:,0]

fig,axes = plt.subplots(nrows=2,figsize=(6,6))

axes[0].plot(motors)
axes[1].plot(U0)

plt.show()
