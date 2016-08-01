import h5py
import numpy as np
import matplotlib.pyplot as plt

fd = h5py.File('FOC_Record.h5', 'r+')
std = fd['/FOC/mox_std'][...]
toa = fd['/FOC/mox_toa'][...]

corr_shift_fl = toa[:,1]
corr_shift_fr = toa[:,2]
corr_shift_lr = toa[:,2] - toa[:,1]

# 2D
R = 5 # 5 cm
loc_f = np.asarray([0, R]) # 5 cm
loc_l = np.asarray([-np.sin(120.0/180.*np.pi)*R, np.cos(120.0/180*np.pi)*R])
loc_r = np.asarray([-np.sin(-120.0/180.*np.pi)*R, np.cos(-120.0/180*np.pi)*R])

psi = np.linspace(-np.pi, np.pi, 36) # 10 degree
e_x = -np.sin(psi)
e_y = np.cos(psi)

# 2D F/L/R time difference
proj_f, proj_l, proj_r = [], [], []
for i in range(len(psi)):
    if psi[i] == -np.pi or psi[i] == np.pi or psi[i] == 0.:
        proj_f.append([0, R])
        proj_l.append([0, loc_l[1]])
        proj_r.append([0, loc_r[1]])
    elif psi[i] == -np.pi/2. or psi[i] == np.pi/2.:
        proj_f.append([0, 0])
        proj_l.append([loc_l[0], 0])
        proj_r.append([loc_r[0], 0])
    else:
        proj_f.append([(loc_f[1]+e_x[i]/e_y[i]*loc_f[0])/(e_y[i]/e_x[i]+e_x[i]/e_y[i]), \
                e_y[i]/e_x[i]*(loc_f[1]+e_x[i]/e_y[i]*loc_f[0])/(e_y[i]/e_x[i]+e_x[i]/e_y[i])])
        proj_l.append([(loc_l[1]+e_x[i]/e_y[i]*loc_l[0])/(e_y[i]/e_x[i]+e_x[i]/e_y[i]), \
                e_y[i]/e_x[i]*(loc_l[1]+e_x[i]/e_y[i]*loc_l[0])/(e_y[i]/e_x[i]+e_x[i]/e_y[i])])
        proj_r.append([(loc_r[1]+e_x[i]/e_y[i]*loc_r[0])/(e_y[i]/e_x[i]+e_x[i]/e_y[i]), \
                e_y[i]/e_x[i]*(loc_r[1]+e_x[i]/e_y[i]*loc_r[0])/(e_y[i]/e_x[i]+e_x[i]/e_y[i])])

v_fl, v_fr, v_lr = [], [], []
for i in range(len(psi)):
    v_fl.append([proj_f[i][0]-proj_l[i][0], proj_f[i][1]-proj_l[i][1]]);
    v_fr.append([proj_f[i][0]-proj_r[i][0], proj_f[i][1]-proj_r[i][1]]);
    v_lr.append([proj_l[i][0]-proj_r[i][0], proj_l[i][1]-proj_r[i][1]]);
s_fl, s_fr, s_lr = [], [], [] # shift
for i in range(len(psi)):
    s_fl.append(v_fl[i][0]*e_x[i] + v_fl[i][1]*e_y[i])
    s_fr.append(v_fr[i][0]*e_x[i] + v_fr[i][1]*e_y[i])
    s_lr.append(v_lr[i][0]*e_x[i] + v_lr[i][1]*e_y[i])

s_fl = np.asarray(s_fl)
s_fr = np.asarray(s_fr)
s_lr = np.asarray(s_lr)

# compute angle
angle = []
for i in range(len(corr_shift_fl)):
    if abs(corr_shift_fl[i]) > 2 or abs(corr_shift_fr[i]) > 2 or abs(corr_shift_lr[i]) > 2:
        angle.append(-3*np.pi)
    else:
        a = np.asarray([corr_shift_fl[i], corr_shift_fr[i], corr_shift_lr[i]])
        cos_angle = []
        for j in range(len(psi)):
            b = np.asarray([s_fl[j], s_fr[j], s_lr[j]])
            cos_angle.append(np.dot(a,b)/np.linalg.norm(a)/np.linalg.norm(b));
        angle.append(psi[np.array(cos_angle).argmin()])

# get original heading
fd_ori = h5py.File('../data/Record_2016-07-08_15-24-37.h5', 'r+')
heading = fd_ori['att_of_robot_0'][...][:,2]
pos = fd_ori['enu_of_robot_0'][...]

fuse_angle = []
for i in range(len(angle)):
    fuse_psi = (angle[i]+heading[i])*180./np.pi
    if fuse_psi > 180:
        fuse_psi = (fuse_psi - 180) - 180;
    elif fuse_psi < -180:
        fuse_psi = (fuse_psi + 180) + 180;
    fuse_angle.append(fuse_psi)

fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4)

ax1.plot(std[:,0], color='r')
ax1.plot(std[:,1], color='y')
ax1.plot(std[:,2], color='b')

ax2.plot(toa[:,0], color='r')
ax2.plot(toa[:,1], color='y')
ax2.plot(toa[:,2], color='b')

ax3.plot(fuse_angle)

ax4.plot(pos[:,1])

plt.show()
