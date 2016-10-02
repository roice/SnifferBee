import h5py
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter

fd = h5py.File("Record_2016-09-27_16-37-07.h5", 'r+')
enu = fd['/robot1/debug/enu'][...]
att = fd['/robot1/debug/att'][...]
vel = fd['/robot1/debug/vel'][...]
acc = fd['/robot1/debug/acc'][...]
vel_p = fd['/robot1/debug/vel_p'][...]
acc_p = fd['/robot1/debug/acc_p'][...]
throttle = fd['/robot1/debug/throttle'][...]
roll = fd['/robot1/debug/roll'][...]
pitch = fd['/robot1/debug/pitch'][...]
yaw = fd['/robot1/debug/yaw'][...]
leso_z1 = fd['/robot1/debug/leso_z1'][...]
leso_z2 = fd['/robot1/debug/leso_z2'][...]
leso_z3 = fd['/robot1/debug/leso_z3'][...]
anemo_1 = fd['/robot1/debug/anemometer_1'][...]
anemo_2 = fd['/robot1/debug/anemometer_2'][...]
anemo_3 = fd['/robot1/debug/anemometer_3'][...]
motors = fd['robot1/motors'][...]
bat_volt = fd['robot1/bat_volt'][...]

leso_z3 = leso_z3[100:]

wind_x = gaussian_filter(leso_z3[:,0], sigma=20)
wind_y = gaussian_filter(leso_z3[:,1], sigma=20)

masked_wind = []
for i in range(len(wind_x)):
    if enu[100+i,1] > -0.9-0.1 and enu[100+i,1] < -0.9+0.1 and enu[100+i,2] > 1.38-0.1 and enu[100+i,2] < 1.38+0.1:
        masked_wind.append([ wind_x[i]-0.15, wind_y[i]-0.06 ])
masked_wind = np.asarray(masked_wind)

theta = []
for i in range(len(masked_wind)):
    temp_theta = math.atan2(-(masked_wind[i,0]-0.16), masked_wind[i,1]-0.08)
    theta.append(temp_theta)
theta = np.asarray(theta)

r = []
for i in range(len(masked_wind)):
    temp_r = math.sqrt(masked_wind[i,0]**2+masked_wind[i,1]**2)
    r.append(temp_r)
r = np.asarray(r)

ax = plt.subplot(111, projection='polar')
ax.scatter(theta, r, s = 10, cmap=plt.cm.hsv)

plt.show()
