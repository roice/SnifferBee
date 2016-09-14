import h5py
import numpy as np
import math
import matplotlib.pyplot as plt

fd = h5py.File("Record_2016-09-13_14-30-24.h5", 'r+')
enu = fd['robot1/debug/enu'][...]
vel = fd['robot1/debug/vel'][...]
acc = fd['robot1/debug/acc'][...]
vel_p = fd['robot1/debug/vel_p'][...]
acc_p = fd['robot1/debug/acc_p'][...]

throttle = fd['robot1/debug/throttle'][...]
roll = fd['robot1/debug/roll'][...]
pitch = fd['robot1/debug/pitch'][...]
yaw = fd['robot1/debug/yaw'][...]

z1 = fd['robot1/debug/leso_z1'][...]
z2 = fd['robot1/debug/leso_z2'][...]
z3 = fd['robot1/debug/leso_z3'][...]

fig, axes = plt.subplots(nrows=5)

'''
axes[0].plot(enu)

axes[1].plot(vel)

axes[2].plot(acc)

axes[3].plot(vel_p)

axes[4].plot(acc_p)
'''


axes[0].plot(throttle)

axes[1].plot(roll)

axes[2].plot(pitch)

axes[3].plot(yaw)


'''
axes[0].plot(z1)

axes[1].plot(z2)

axes[2].plot(z3)
'''

plt.show()
