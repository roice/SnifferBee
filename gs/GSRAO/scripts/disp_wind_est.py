import h5py
import numpy as np
import matplotlib.pyplot as plt

fd = h5py.File('./Record_2017-03-28_23-11-42.h5', 'r+')

c = fd['robot1/debug/wind_resist_coef']
wind = fd['robot1/debug/wind_estimated']

fig,axes = plt.subplots(nrows=2,figsize=(6,6))

axes[0].plot(c)

plt.show()
