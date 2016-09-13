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


fd = h5py.File("Record_2016-09-13_11-49-20.h5", 'r+')
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
anemo_1 = fd['/anemometers/1'][...]
anemo_2 = fd['/anemometers/2'][...]
anemo_3 = fd['/anemometers/3'][...]

# interpolate anemo to the same length of vel_p
t_50Hz = np.linspace(0, 10, len(vel_p))
t_32Hz = np.linspace(0, 10, len(anemo_1))
f = interpolate.interp1d(t_32Hz, anemo_1[:,0], kind='slinear')
anemo_1_x = f(t_50Hz)
f = interpolate.interp1d(t_32Hz, anemo_1[:,1], kind='slinear')
anemo_1_y = f(t_50Hz)
f = interpolate.interp1d(t_32Hz, anemo_1[:,2], kind='slinear')
anemo_1_z = f(t_50Hz)
t_32Hz = np.linspace(0, 10, len(anemo_2))
f = interpolate.interp1d(t_32Hz, anemo_2[:,0], kind='slinear')
anemo_2_x = f(t_50Hz)
f = interpolate.interp1d(t_32Hz, anemo_2[:,1], kind='slinear')
anemo_2_y = f(t_50Hz)
f = interpolate.interp1d(t_32Hz, anemo_2[:,2], kind='slinear')
anemo_2_z = f(t_50Hz)
t_32Hz = np.linspace(0, 10, len(anemo_3))
f = interpolate.interp1d(t_32Hz, anemo_3[:,0], kind='slinear')
anemo_3_x = f(t_50Hz)
f = interpolate.interp1d(t_32Hz, anemo_3[:,1], kind='slinear')
anemo_3_y = f(t_50Hz)
f = interpolate.interp1d(t_32Hz, anemo_3[:,2], kind='slinear')
anemo_3_z = f(t_50Hz)

# interpolate anemo to get wind speed at robot's position
anemo = []
pos_anemo_1 = np.asarray([ 0, -1.8, 1.3 ])
pos_anemo_2 = np.asarray([ 1.0, 0, 1.3 ])
pos_anemo_3 = np.asarray([ -1.0, 0, 1.3 ])
for i in range(len(vel_p)):
    w1 = math.sqrt((enu[i,0]-pos_anemo_1[0])**2 + (enu[i,1]-pos_anemo_1[1])**2 + (enu[i,2]-pos_anemo_1[2])**2)
    w2 = math.sqrt((enu[i,0]-pos_anemo_2[0])**2 + (enu[i,1]-pos_anemo_2[1])**2 + (enu[i,2]-pos_anemo_2[2])**2)
    w3 = math.sqrt((enu[i,0]-pos_anemo_3[0])**2 + (enu[i,1]-pos_anemo_3[1])**2 + (enu[i,2]-pos_anemo_3[2])**2)
    sum_w = w1 + w2 + w3
    w1 /= sum_w
    w2 /= sum_w
    w3 /= sum_w
    temp_anemo_x = anemo_1_x[i]*w1 + anemo_2_x[i]*w2 + anemo_3_x[i]*w3
    temp_anemo_y = anemo_1_y[i]*w1 + anemo_2_y[i]*w2 + anemo_3_y[i]*w3
    temp_anemo_z = anemo_1_z[i]*w1 + anemo_2_z[i]*w2 + anemo_3_z[i]*w3
    anemo.append([ temp_anemo_x, temp_anemo_y, temp_anemo_z ])
anemo = np.asarray(anemo)

# LESO
dt = 1.0/50.0   # 50 Hz
w0 = 18.0
temp_z1 = 0.
temp_z2 = 0.
temp_z3 = 0.
z1_roll_p = []
z2_roll_p = []
z3_roll_p = []
for i in range(len(vel_p)):
    leso_err = vel_p[i,0] - temp_z1
    temp_z1 += dt*(temp_z2 + 3*w0*leso_err)
    temp_z2 += dt*(temp_z3 + 3*pow(w0, 2)*leso_err + roll[i,0])
    temp_z3 += dt*(pow(w0, 3)*leso_err)
    z1_roll_p.append(temp_z1)
    z2_roll_p.append(temp_z2)
    z3_roll_p.append(temp_z3)
z1_roll_p = np.asarray(z1_roll_p)
z2_roll_p = np.asarray(z2_roll_p)
z3_roll_p = np.asarray(z3_roll_p)

temp_z1 = 0.
temp_z2 = 0.
temp_z3 = 0.
z1_pitch_p = []
z2_pitch_p = []
z3_pitch_p = []
for i in range(len(vel_p)):
    leso_err = vel_p[i,1] - temp_z1
    temp_z1 += dt*(temp_z2 + 3*w0*leso_err)
    temp_z2 += dt*(temp_z3 + 3*pow(w0, 2)*leso_err + pitch[i,0])
    temp_z3 += dt*(pow(w0, 3)*leso_err)
    z1_pitch_p.append(temp_z1)
    z2_pitch_p.append(temp_z2)
    z3_pitch_p.append(temp_z3)
z1_pitch_p = np.asarray(z1_pitch_p)
z2_pitch_p = np.asarray(z2_pitch_p)
z3_pitch_p = np.asarray(z3_pitch_p)

temp_z1 = 0.
temp_z2 = 0.
temp_z3 = 0.
z1_throttle_p = []
z2_throttle_p = []
z3_throttle_p = []
for i in range(len(vel_p)):
    leso_err = vel_p[i,2] - temp_z1
    temp_z1 += dt*(temp_z2 + 3*w0*leso_err)
    temp_z2 += dt*(temp_z3 + 3*pow(w0, 2)*leso_err + throttle[i,0])
    temp_z3 += dt*(pow(w0, 3)*leso_err)
    z1_throttle_p.append(temp_z1)
    z2_throttle_p.append(temp_z2)
    z3_throttle_p.append(temp_z3)
z1_throttle_p = np.asarray(z1_throttle_p)
z2_throttle_p = np.asarray(z2_throttle_p)
z3_throttle_p = np.asarray(z3_throttle_p)

# convert z3 from plane coordinate to earth coordinate
z3 = []
for i in range(len(vel_p)):
    v_p = np.asarray([z3_roll_p[i], z3_pitch_p[i], z3_throttle_p[i]])
    out = rotate_vector(v_p, att[i,2], 0, 0)
    z3.append(out)
z3 = np.asarray(z3)

alpha = 0.003

fig, axes = plt.subplots(nrows=6)

axes[0].plot(vel_p[:,0], color = 'red')
axes[0].plot(z1_roll_p, color = 'blue')

axes[1].plot(vel_p[:,1], color = 'red')
axes[1].plot(z1_pitch_p, color = 'blue')

axes[2].plot(vel_p[:,2], color = 'red')
axes[2].plot(z1_throttle_p, color = 'blue')

axes[3].plot(-anemo[:,0], color = 'red')
axes[3].plot(alpha*z3[:,0], color = 'blue')

axes[4].plot(-anemo[:,1], color = 'red')
axes[4].plot(alpha*z3[:,1], color = 'blue')

axes[5].plot(-anemo[:,2], color = 'red')
axes[5].plot(alpha*z3[:,2], color = 'blue')

plt.show()
