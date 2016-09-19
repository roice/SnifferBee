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


fd = h5py.File("Record_2016-09-19_22-46-55.h5", 'r+')
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

'''
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
'''

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
    temp_anemo_x = anemo_1[i,0]*w1 + anemo_2[i,0]*w2 + anemo_3[i,0]*w3
    temp_anemo_y = anemo_1[i,1]*w1 + anemo_2[i,1]*w2 + anemo_3[i,1]*w3
    temp_anemo_z = anemo_1[i,2]*w1 + anemo_2[i,2]*w2 + anemo_3[i,2]*w3
    anemo.append([ temp_anemo_x, temp_anemo_y, temp_anemo_z ])
anemo = np.asarray(anemo)

roll_pitch = []
for i in range(len(roll)):
    temp_v = rotate_vector(np.asarray([ roll[i,0], pitch[i,0], 0. ]), att[i,0], 0., 0.)
    roll_pitch.append(temp_v)
roll_pitch = np.asarray(roll_pitch)

delay = 0.0    # s
delayed_roll_pitch = []
for i in range(len(roll_pitch)):
    if (i < int(delay*50.0)):
        delayed_roll_pitch.append([0., 0.])
    else:
        delayed_roll_pitch.append([ roll_pitch[i-int(delay*50.0),0], roll_pitch[i-int(delay*50.0),1] ])
delayed_roll_pitch = np.asarray(delayed_roll_pitch)


# LESO
scale_u = 0.03
dt = 1.0/50.0   # 50 Hz
w0 = 10.0
temp_z1 = 0.
temp_z2 = 0.
temp_z3 = 0.
z1_roll = []
z2_roll = []
z3_roll = []
for i in range(len(enu)):
    leso_err = enu[i,0] - temp_z1
    temp_z1 += dt*(temp_z2 + 3*w0*leso_err)
    temp_z2 += dt*(temp_z3 + 3*pow(w0, 2)*leso_err + scale_u*roll_pitch[i,0])
    temp_z3 += dt*(pow(w0, 3)*leso_err)
    z1_roll.append(temp_z1)
    z2_roll.append(temp_z2)
    z3_roll.append(temp_z3)
z1_roll = np.asarray(z1_roll)
z2_roll = np.asarray(z2_roll)
z3_roll = np.asarray(z3_roll)

temp_z1 = 0.
temp_z2 = 0.
temp_z3 = 0.
z1_pitch = []
z2_pitch = []
z3_pitch = []
for i in range(len(enu)):
    leso_err = enu[i,1] - temp_z1
    temp_z1 += dt*(temp_z2 + 3*w0*leso_err)
    temp_z2 += dt*(temp_z3 + 3*pow(w0, 2)*leso_err + scale_u*delayed_roll_pitch[i,1])
    temp_z3 += dt*(pow(w0, 3)*leso_err)
    z1_pitch.append(temp_z1)
    z2_pitch.append(temp_z2)
    z3_pitch.append(temp_z3)
z1_pitch = np.asarray(z1_pitch)
z2_pitch = np.asarray(z2_pitch)
z3_pitch = np.asarray(z3_pitch)

temp_z1 = 0.
temp_z2 = 0.
temp_z3 = 0.
z1_throttle = []
z2_throttle = []
z3_throttle = []
for i in range(len(enu)):
    leso_err = enu[i,2] - temp_z1
    temp_z1 += dt*(temp_z2 + 3*w0*leso_err)
    temp_z2 += dt*(temp_z3 + 3*pow(w0, 2)*leso_err + scale_u*throttle[i,0])
    temp_z3 += dt*(pow(w0, 3)*leso_err)
    z1_throttle.append(temp_z1)
    z2_throttle.append(temp_z2)
    z3_throttle.append(temp_z3)
z1_throttle = np.asarray(z1_throttle)
z2_throttle = np.asarray(z2_throttle)
z3_throttle = np.asarray(z3_throttle)

direction = []
for i in range(len(z3_roll)):
    direction.append(math.atan2(z3_pitch[i], z3_roll[i]))
direction = np.asarray(direction)

factor_wind_acc = 6.0

print "factor of wind vs vel is " + str(np.std(z3_roll[100:])/np.std(vel[100:,0]))
print "factor of wind vs vel is " + str(np.std(z3_pitch[100:])/np.std(vel[100:,1]))

fig, axes = plt.subplots(nrows=2)

'''
axes[0].plot(vel[:,1], color = 'red')
axes[0].plot(z3_pitch/factor_wind_acc, color = 'blue')
axes[0].plot(vel[:,1]-z3_pitch/factor_wind_acc, color = 'k')
#axes[0].plot(scale_u*roll_pitch[:,1], color = 'green')
'''


'''
axes[0].plot(enu[:,0], color = 'red')
axes[0].plot(z1_roll, color = 'blue')

axes[1].plot(vel[:,1], color = 'red')
axes[1].plot(z1_pitch, color = 'blue')
axes[1].plot(z3_pitch/factor_wind_acc, color = 'green')

axes[2].plot(enu[:,2], color = 'red')
axes[2].plot(z1_throttle, color = 'blue')
'''

'''
axes[3].plot(vel[:,0], color = 'red')
axes[3].plot(z2_roll, color = 'blue')

axes[4].plot(vel[:,1], color = 'red')
axes[4].plot(z2_pitch, color = 'blue')

axes[5].plot(vel[:,2], color = 'red')
axes[5].plot(z2_throttle, color = 'blue')
'''


'''
axes[3].plot(anemo[:,0], color = 'red')
axes[3].plot(z3_roll/factor_wind_acc, color = 'blue')
#axes[3].plot(z3_roll, color = 'blue')
'''


axes[0].plot(anemo[100:,1], color = 'red')
axes[0].plot(z3_pitch[100:]/factor_wind_acc, color = 'blue')
#axes[0].plot(0.001*roll_pitch[100:,1], color = 'green')
#axes[0].plot(0.001*delayed_roll_pitch[100:,1], color = 'k')

'''
axes[5].plot(anemo[:,2], color = 'red')
axes[5].plot(z3_throttle/factor_wind_acc, color = 'blue')
#axes[5].plot(z3_throttle, color = 'blue')
#axes[5].plot(direction, color = 'blue')
'''

plt.show()
