import h5py
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.ndimage.filters import gaussian_filter

fd = h5py.File("Record_2016-09-30_15-24-06.h5", 'r+')
enu = fd['robot1/debug/enu'][...]
vel = fd['robot1/debug/vel'][...]
acc = fd['robot1/debug/acc'][...]
vel_p = fd['robot1/debug/vel_p'][...]
acc_p = fd['robot1/debug/acc_p'][...]
att = fd['robot1/debug/att'][...]

throttle = fd['robot1/debug/throttle'][...]
roll = fd['robot1/debug/roll'][...]
pitch = fd['robot1/debug/pitch'][...]
yaw = fd['robot1/debug/yaw'][...]

wind = fd['robot1/wind'][...]

motors = fd['robot1/motors'][...]

z1 = fd['robot1/debug/leso_z1'][...]
z2 = fd['robot1/debug/leso_z2'][...]
z3 = fd['robot1/debug/leso_z3'][...]

bat_volt = fd['robot1/bat_volt'][...]

anemo_2 = fd['/robot1/debug/anemometer_2'][...]



# interpolate motors & bat volt to the same length of vel
t_50Hz = np.linspace(0, 10, len(vel))
t_20Hz = np.linspace(0, 10, len(motors))
f = interpolate.interp1d(t_20Hz, motors[:,0], kind='slinear')
motor1 = f(t_50Hz)
f = interpolate.interp1d(t_20Hz, motors[:,1], kind='slinear')
motor2 = f(t_50Hz)
f = interpolate.interp1d(t_20Hz, motors[:,2], kind='slinear')
motor3 = f(t_50Hz)
f = interpolate.interp1d(t_20Hz, motors[:,3], kind='slinear')
motor4 = f(t_50Hz)


pos_x = gaussian_filter(enu[:,0], sigma=10)
pos_y = gaussian_filter(enu[:,1], sigma=10)
pos_z = gaussian_filter(enu[:,2], sigma=10)

vel_x = gaussian_filter(vel[:,0], sigma=10)
vel_y = gaussian_filter(vel[:,1], sigma=10)
vel_z = gaussian_filter(vel[:,2], sigma=10)

att_x = gaussian_filter(att[:,0], sigma=10)
att_y = gaussian_filter(att[:,1], sigma=10)
att_z = gaussian_filter(att[:,2], sigma=10)

motor_1 = gaussian_filter(motor1, sigma=10)
motor_2 = gaussian_filter(motor2, sigma=10)
motor_3 = gaussian_filter(motor3, sigma=10)
motor_4 = gaussian_filter(motor4, sigma=10)


print "self disturbance mean: [ " + str(np.mean(wind[100:,0])) + "," + str(np.mean(wind[100:,1])) + " ]"

fig, axes = plt.subplots(nrows=5)

#axes[0].plot(anemo_2)
#axes[0].plot(wind[100:,0], color = 'green')
#axes[0].plot(wind[100:,1], color = 'blue')
axes[0].plot(enu[:,0], color = 'red')
#axes[0].plot(pos_x, color = 'blue')

'''
axes[1].plot(z3[100:,0], color = 'green')
axes[1].plot(z3[100:,1], color = 'blue')
'''

axes[1].plot(vel[:,0], color = 'red')
axes[1].plot(vel_x, color = 'blue')

'''
axes[2].plot(motors[:,0], color = 'red')
axes[2].plot(motor_1, color = 'blue')

axes[3].plot(att[:,0], color = 'red')
axes[3].plot(att_x, color = 'blue')
'''



'''
axes[0].plot(throttle)

axes[1].plot(roll)

axes[2].plot(pitch)

axes[3].plot(yaw)
'''

'''
axes[0].plot(enu[:,0], color = 'red')
axes[0].plot(z1[:,0], color = 'blue')

axes[1].plot(enu[:,1], color = 'red')
axes[1].plot(z1[:,1], color = 'blue')

axes[2].plot(z3[:,0], color = 'green')
axes[2].plot(z3[:,1], color = 'blue')
'''

plt.show()
