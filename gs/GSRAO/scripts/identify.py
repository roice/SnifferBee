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


fd = h5py.File("Record_2016-09-25_16-42-59.h5", 'r+')
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

pos_x = gaussian_filter(enu[:,0], sigma=10)
pos_y = gaussian_filter(enu[:,1], sigma=10)
pos_z = gaussian_filter(enu[:,2], sigma=10)

vel_x = gaussian_filter(vel[:,0], sigma=10)
vel_y = gaussian_filter(vel[:,1], sigma=10)
vel_z = gaussian_filter(vel[:,2], sigma=10)

acc_x = gaussian_filter(acc[:,0], sigma=10)
acc_y = gaussian_filter(acc[:,1], sigma=10)
acc_z = gaussian_filter(acc[:,2], sigma=10)

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

u = []
for i in range(len(motor_1)):
    temp_motors = rotate_vector(np.asarray([ 0., 0., (motor_1[i]**2+motor_2[i]**2+motor_3[i]**2+motor_4[i]**2)]), att_z[i], att_y[i], att_x[i])
    scale_m_to_u = (acc_z[i] + 9.8)/temp_motors[2]
    temp_u = scale_m_to_u*temp_motors
    u.append(temp_u)
u = np.asarray(u)

# LESO
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
    temp_z2 += dt*(temp_z3 + 3*pow(w0, 2)*leso_err + u[i,0])
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
    temp_z2 += dt*(temp_z3 + 3*pow(w0, 2)*leso_err + u[i,1])
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
    temp_z2 += dt*(temp_z3 + 3*pow(w0, 2)*leso_err + u[i,2]-9.8)
    temp_z3 += dt*(pow(w0, 3)*leso_err)
    z1_z.append(temp_z1)
    z2_z.append(temp_z2)
    z3_z.append(temp_z3)
z1_z = np.asarray(z1_z)
z2_z = np.asarray(z2_z)
z3_z = np.asarray(z3_z)

factor_z3_to_wind = 1.0

# calculate dx/dy/dz
d_x = []
d_y = []
d_z = []
for i in range(len(z3_x)):
    yaw = att_z[i]
    pitch = att_y[i]
    roll = att_x[i]
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
    z3_xyz = np.asarray([ z3_x[i], z3_y[i], z3_z[i] ])
    d_xyz = np.linalg.solve(R_zyx, z3_xyz)
    d_x.append(d_xyz[0])
    d_y.append(d_xyz[1])
    d_z.append(d_xyz[2])
d_x = np.asarray(d_x)
d_y = np.asarray(d_y)
d_z = np.asarray(d_z)

print "d_xyz = [ " + str(np.mean(d_x[400:3400])) + ", " + str(np.mean(d_y[400:3400])) + ", " + str(np.mean(d_z[400:3400])) + " ]"

'''
disturb_x = np.mean(d_x[400:3400])
disturb_y = np.mean(d_y[400:3400])
disturb_z = np.mean(d_z[400:3400])
'''

disturb_x = 0.0927989075586
disturb_y = -0.00826646558869
disturb_z = -0.00428738381195

d_i_x = []
d_i_y = []
d_i_z = []
for i in range(len(att)):
    yaw = att_z[i]
    pitch = att_y[i]
    roll = att_x[i]
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
    d_i_xyz = np.dot(R_zyx, np.asarray([disturb_x, disturb_y, disturb_z]))
    d_i_x.append(d_i_xyz[0])
    d_i_y.append(d_i_xyz[1])
    d_i_z.append(d_i_xyz[2])
d_i_x = np.asarray(d_i_x)
d_i_y = np.asarray(d_i_y)
d_i_z = np.asarray(d_i_z)

# interpolate wind to the same length of vel, down sampling
t_50Hz = np.linspace(0, 10, len(vel))
t_20Hz = np.linspace(0, 10, len(motors))
f = interpolate.interp1d(t_50Hz, z3_x-disturb_x, kind='slinear')
wind_x = f(t_20Hz)
f = interpolate.interp1d(t_50Hz, z3_y-disturb_y, kind='slinear')
wind_y = f(t_20Hz)
f = interpolate.interp1d(t_50Hz, z3_z-disturb_z, kind='slinear')
wind_z = f(t_20Hz)

fd = h5py.File("wind.h5", "w")
dset = fd.create_dataset("wind", (len(t_20Hz), 3), dtype='float')
dset[:,0] = wind_x
dset[:,1] = wind_y
dset[:,2] = wind_z

fig, axes = plt.subplots(nrows=3)

axes[0].plot(att_x, color = 'red')
axes[0].plot(att_y, color = 'blue')
axes[0].plot(att_z, color = 'green')

axes[1].plot(z3_x[400:]-disturb_x, color = 'red')
axes[1].plot(z3_y[400:]-disturb_y, color = 'blue')
#axes[1].plot(d_i_y[400:], color = 'blue')
#axes[1].plot(z3_y[400:] - d_i_y[400:], color = 'green')

axes[2].plot(u[:,0], color = 'red')
axes[2].plot(u[:,1], color = 'blue')
axes[2].plot(u[:,2], color = 'green')

plt.show()
