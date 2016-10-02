import h5py
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.ndimage.filters import gaussian_filter

def rotate_vector(vector, yaw, pitch, roll):
    """docstring for rotate_vector"""
    sin_yaw = math.sin(yaw)
    cos_yaw = math.cos(yaw)
    sin_pitch = math.sin(pitch)
    cos_pitch = math.cos(pitch)
    sin_roll = math.sin(roll)
    cos_roll = math.cos(roll)
    R_z = np.asarray([ [cos_yaw, -sin_yaw, 0.], [sin_yaw, cos_yaw, 0.], [0., 0., 1.] ])
    R_y = np.asarray([ [cos_pitch, 0., sin_pitch], [0., 1., 0.], [-sin_pitch, 0., cos_pitch] ])
    R_x = np.asarray([ [1., 0., 0.], [0., cos_roll, -sin_roll], [0., sin_roll, cos_roll] ])
    R_zyx = np.dot(np.dot(R_z, R_y), R_x)
    out = np.dot(R_zyx, vector)
    return out


fd = h5py.File("Record_2016-10-02_11-30-09.h5", 'r+')
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
filtered_bat_volt = gaussian_filter(bat_volt[:,0], sigma=10)
f = interpolate.interp1d(t_20Hz, filtered_bat_volt, kind='slinear')
bat_volt = f(t_50Hz)


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
'''
pos_x = enu[:,0]
pos_y = enu[:,1]
pos_z = enu[:,2]

vel_x = vel[:,0]
vel_y = vel[:,1]
vel_z = vel[:,2]

att_x = att[:,0]
att_y = att[:,1]
att_z = att[:,2]

motor_1 = motor1
motor_2 = motor2
motor_3 = motor3
motor_4 = motor4
'''

# interpolate anemo to get wind speed at robot's position
anemo = []
pos_anemo_1 = np.asarray([ 0, -1.8, 1.38 ])
pos_anemo_2 = np.asarray([ 1.0, 0, 1.38 ])
pos_anemo_3 = np.asarray([ -1.0, 0, 1.38 ])


for i in range(len(vel_p)):
    w1 = 1.0/math.sqrt((enu[i,0]-pos_anemo_1[0])**2 + (enu[i,1]-pos_anemo_1[1])**2 + (enu[i,2]-pos_anemo_1[2])**2)
    w2 = 1.0/math.sqrt((enu[i,0]-pos_anemo_2[0])**2 + (enu[i,1]-pos_anemo_2[1])**2 + (enu[i,2]-pos_anemo_2[2])**2)
    w3 = 1.0/math.sqrt((enu[i,0]-pos_anemo_3[0])**2 + (enu[i,1]-pos_anemo_3[1])**2 + (enu[i,2]-pos_anemo_3[2])**2)
    sum_w = w1 + w2 + w3
    w1 /= sum_w
    w2 /= sum_w
    w3 /= sum_w
    temp_anemo_x = anemo_1[i,0]*w1 + anemo_2[i,0]*w2 + anemo_3[i,0]*w3
    temp_anemo_y = anemo_1[i,1]*w1 + anemo_2[i,1]*w2 + anemo_3[i,1]*w3
    temp_anemo_z = anemo_1[i,2]*w1 + anemo_2[i,2]*w2 + anemo_3[i,2]*w3
    anemo.append([ temp_anemo_x, temp_anemo_y, temp_anemo_z ])
anemo = np.asarray(anemo)

u = []
for i in range(len(motor_1)):
    temp_u = rotate_vector(np.asarray([ 0., 0., (motor_1[i]+motor_2[i]+motor_3[i]+motor_4[i])]), att_z[i], att_y[i], att_x[i])
    u.append(temp_u)
u = np.asarray(u)

# LESO
scale_u = 0.002
dt = 1.0/50.0   # 50 Hz
w0 = 10.0
temp_z1 = 0.
temp_z2 = 0.
temp_z3 = 0.
z1_x = []
z2_x = []
z3_x = []
for i in range(len(enu)):
    leso_err = pos_x[i] - temp_z1
    temp_z1 += dt*(temp_z2 + 3*w0*leso_err)
    temp_z2 += dt*(temp_z3 + 3*pow(w0, 2)*leso_err + scale_u*u[i,0])
    temp_z3 += dt*(pow(w0, 3)*leso_err)
    z1_x.append(temp_z1)
    z2_x.append(temp_z2)
    z3_x.append(temp_z3)
z1_x = np.asarray(z1_x)
z2_x = np.asarray(z2_x)
z3_x = np.asarray(z3_x)


temp_z1 = 0.
temp_z2 = 0.
temp_z3 = 0.
z1_y = []
z2_y = []
z3_y = []
for i in range(len(enu)):
    leso_err = pos_y[i] - temp_z1
    temp_z1 += dt*(temp_z2 + 3*w0*leso_err)
    temp_z2 += dt*(temp_z3 + 3*pow(w0, 2)*leso_err + scale_u*u[i,1])
    temp_z3 += dt*(pow(w0, 3)*leso_err)
    z1_y.append(temp_z1)
    z2_y.append(temp_z2)
    z3_y.append(temp_z3)
z1_y = np.asarray(z1_y)
z2_y = np.asarray(z2_y)
z3_y = np.asarray(z3_y)

temp_z1 = 0.
temp_z2 = 0.
temp_z3 = 0.
z1_z = []
z2_z = []
z3_z = []
for i in range(len(enu)):
    leso_err = pos_z[i] - temp_z1
    temp_z1 += dt*(temp_z2 + 3*w0*leso_err)
    temp_z2 += dt*(temp_z3 + 3*pow(w0, 2)*leso_err + scale_u*u[i,2])
    temp_z3 += dt*(pow(w0, 3)*leso_err)
    z1_z.append(temp_z1)
    z2_z.append(temp_z2)
    z3_z.append(temp_z3)
z1_z = np.asarray(z1_z)
z2_z = np.asarray(z2_z)
z3_z = np.asarray(z3_z)

factor_z3_to_wind = 1.0

wind = []
for i in range(len(z3_x)):
    temp_wind = [ z3_x[i]*factor_z3_to_wind, z3_y[i]*factor_z3_to_wind, z3_z[i]*factor_z3_to_wind ]
    if math.sqrt(temp_wind[0]**2+temp_wind[1]**2) < 0:
        temp_wind = [ 0., 0., 0. ]
    wind.append(temp_wind)
wind = np.asarray(wind)

# convert wind from earth coord to body coord
wind_p = []
for i in range(len(wind)):
    temp_wind_p = rotate_vector(wind[i], att[i,2], 0, 0)
    wind_p.append(temp_wind_p)
wind_p = np.asarray(wind_p)

filtered_wind_x = gaussian_filter(wind[100:,0], sigma=50)
filtered_anemo_x = gaussian_filter(anemo[100:,0], sigma=50)
filtered_wind_y = gaussian_filter(wind[100:,1], sigma=50)
filtered_anemo_y = gaussian_filter(anemo[100:,1], sigma=50)
filtered_wind_z = gaussian_filter(wind[100:,2], sigma=50)
filtered_anemo_z = gaussian_filter(anemo[100:,2], sigma=50)


'''
masked_wind = []
for i in range(len(filtered_wind_x)):
    if enu[100+i,1] > -0.9-0.05 and enu[100+i,1] < -0.9+0.05 and enu[100+i,2] > 1.38-0.05 and enu[100+i,2] < 1.38+0.05:
        masked_wind.append([ filtered_wind_x[i]-0.15, filtered_wind_y[i]-0.06 ])
masked_wind = np.asarray(masked_wind)
'''

'''
theta = []
r = []
for i in range(len(wind)):
    temp_theta = math.atan2(-(wind[i,0]-0.16), wind[i,1]-0.08)
    temp_r = math.sqrt(wind[i,0]**2+wind[i,1]**2)
    if temp_r < 1.0:
        r.append(temp_r)
        theta.append(temp_theta)
theta = np.asarray(theta)
r = np.asarray(r)

ax = plt.subplot(111, projection='polar')
ax.scatter(theta, r, s = 10, cmap=plt.cm.hsv)
'''



fig, axes = plt.subplots(nrows=3)

#axes[0].plot((motor_1+motor_2+motor_3+motor_4))
#axes[0].plot(u[:,0])
axes[0].plot(pos_z, color = 'red')
axes[0].plot(z1_z, color = 'blue')
#axes[0].plot(att[:,2], color = 'blue')

#axes[1].plot(vel_y, color = 'red')
axes[1].plot(att_x, color = 'blue')

#axes[2].plot(anemo[100:,0], color = 'red')
axes[2].plot(wind[100:,1], color = 'red')
#axes[2].plot(filtered_anemo_x, color = 'red')
#axes[2].plot(filtered_wind_x-0.15, color = 'blue')
#axes[2].plot(leso_z3[100:,0], color = 'green')
#axes[2].plot(leso_z3[100:,1], color = 'k')
#axes[2].plot(filtered_wind_x, color = 'green')
#axes[2].plot(filtered_wind_y, color = 'k')
#axes[2].plot(masked_wind[:,0], color = 'green')
#axes[2].plot(masked_wind[:,1], color = 'k')



'''
axes[3].plot(vel[:,0], color = 'red')
axes[3].plot(z2_roll, color = 'blue')

axes[4].plot(vel[:,1], color = 'red')
axes[4].plot(z2_pitch, color = 'blue')

axes[5].plot(vel[:,2], color = 'red')
axes[5].plot(z2_throttle, color = 'blue')
'''


'''
filtered_anemo = gaussian_filter(anemo[100:,1], sigma=100)
filtered_z3 = gaussian_filter(z3_pitch[100:]/factor_wind_acc, sigma=100)

axes[0].plot(anemo[100:,1], color = 'red')
axes[0].plot(z3_pitch[100:]/factor_wind_acc, color = 'blue')
#axes[0].plot(filtered_anemo, color = 'red')
#axes[0].plot(filtered_z3, color = 'blue')
'''

'''
axes[5].plot(anemo[:,2], color = 'red')
axes[5].plot(z3_throttle/factor_wind_acc, color = 'blue')
#axes[5].plot(z3_throttle, color = 'blue')
#axes[5].plot(direction, color = 'blue')
'''

plt.show()
