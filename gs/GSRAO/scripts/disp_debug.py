import h5py
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter

fd = h5py.File("Record_2016-09-18_19-16-58.h5", 'r+')
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

acc_pitch_p = gaussian_filter(acc_p[:,1], sigma=10)

print "roll factor = " + str(np.std(acc_p[:,0])/np.std(roll))
print "pitch factor = " + str(np.std(acc_p[:,1])/np.std(pitch))

fig, axes = plt.subplots(nrows=5)

'''
axes[0].plot(enu)

axes[1].plot(vel)

axes[2].plot(acc)

axes[3].plot(vel_p)

axes[4].plot(acc_p)
'''


axes[0].plot(throttle)

axes[1].plot(0.03*roll, color = 'red')
axes[1].plot(acc_p[:,0], color = 'blue')

axes[2].plot(0.03*pitch, color = 'red')
axes[2].plot(acc_pitch_p, color = 'blue')

axes[3].plot(yaw)


'''
axes[0].plot(enu[:,0], color = 'red')
axes[0].plot(z1[:,0], color = 'blue')

axes[1].plot(enu[:,1], color = 'red')
axes[1].plot(z1[:,1], color = 'blue')

axes[2].plot(z3[:,0], color = 'green')
axes[2].plot(z3[:,1], color = 'blue')
'''

plt.show()
