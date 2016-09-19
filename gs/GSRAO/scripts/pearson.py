import h5py
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import interpolate

def multipl(a,b):
    sumofab=0.0
    for i in range(len(a)):
        temp=a[i]*b[i]
        sumofab+=temp
    return sumofab

def corrcoef(x,y):
    n=len(x)
    sum1=sum(x)
    sum2=sum(y)
    sumofxy=multipl(x,y)
    sumofx2 = sum([pow(i,2) for i in x])
    sumofy2 = sum([pow(j,2) for j in y])
    num=sumofxy-(float(sum1)*float(sum2)/n)
    den=math.sqrt((sumofx2-float(sum1**2)/n)*(sumofy2-float(sum2**2)/n))
    return num/den

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

fd = h5py.File("Record_2016-09-18_17-09-55.h5", 'r+')
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
    temp_z2 += dt*(temp_z3 + 3*pow(w0, 2)*leso_err + scale_u*roll_pitch[i,1])
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


alpha_list = []
coef_list = []
for temp_alpha in np.linspace(0.01, 1.0, 100):
    alpha_list.append(temp_alpha)
    coef_list.append(0.5*corrcoef(temp_alpha*z3_roll, anemo[:,0])+0.5*corrcoef(temp_alpha*z3_pitch, anemo[:,1]))
    #coef_list.append(0.5*corrcoef(temp_alpha*z3[:,0], anemo[:,0])+0.5*corrcoef(temp_alpha*z3[:,1], anemo[:,1]))

print alpha_list[coef_list.index(min(coef_list))]

fig, axes = plt.subplots(nrows=2)

axes[0].plot(alpha_list, coef_list)

plt.show()
