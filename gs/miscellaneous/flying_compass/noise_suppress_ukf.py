import h5py
import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import UnscentedKalmanFilter
from filterpy.kalman import MerweScaledSigmaPoints as SigmaPoints
from filterpy.common import Q_discrete_white_noise

# get signal
fd_record = h5py.File('Record_2016-06-07_22-23-42.h5', 'r+')
time = fd_record['time_of_robot_0'][...]
signal_front = fd_record['sensors_of_robot_0'][...][:,0]
signal_left = fd_record['sensors_of_robot_0'][...][:,1]
signal_right = fd_record['sensors_of_robot_0'][...][:,2]
# inverse signal, high voltage correspond to high concentration
signal_front = 3.3 - signal_front
signal_left = 3.3 - signal_left
signal_right = 3.3 - signal_right

# state transition function and measurement function
alpha = 0.5
def f_mox(x, dt):
    F = np.array([[0, -1./alpha, 1,  0],
                  [0,  1,        0,  0],
                  [0,  0,        1, dt],
                  [0,  0,        0,  1]])
    return np.dot(F, x)

def h_mox(x):
    return np.array([x[0]])

# Unscented Kalman Filter
def sensor_ukf(dim_x, dim_z, fx, hx, dt, points):
    kf = UnscentedKalmanFilter(dim_x, dim_z, dt, fx=fx, hx=hx, points=points)

    kf.Q[0:2, 0:2] = Q_discrete_white_noise(2, dt=dt, var=0.01)
    kf.Q[2:4, 2:4] = Q_discrete_white_noise(2, dt=dt, var=0.01)

    kf.R = np.array([ [0.05**2] ])
    kf.x = np.zeros(4)
    kf.P = np.eye(4) * 500.
    return kf

# init filter
sigmas_front = SigmaPoints(n=4, alpha=.1, beta=2., kappa=1.)
sigmas_left = SigmaPoints(n=4, alpha=.1, beta=2., kappa=1.)
sigmas_right = SigmaPoints(n=4, alpha=.1, beta=2., kappa=1.)
kf_front = sensor_ukf(dim_x=4, dim_z=1, fx=f_mox,
        hx=h_mox, dt=0.1, points=sigmas_front)
kf_left = sensor_ukf(dim_x=4, dim_z=1, fx=f_mox,
        hx=h_mox, dt=0.1, points=sigmas_left)
kf_right = sensor_ukf(dim_x=4, dim_z=1, fx=f_mox,
        hx=h_mox, dt=0.1, points=sigmas_right)
# run
mu_front, cov_front = [], []
mu_left, cov_left = [], []
mu_right, cov_right = [], []
for i in range(len(signal_front)):
    kf_front.predict()
    kf_front.update([signal_front[i]])
    mu_front.append(kf_front.x)
    cov_front.append(kf_front.P)
    kf_left.predict()
    kf_left.update([signal_left[i]])
    mu_left.append(kf_left.x)
    cov_left.append(kf_left.P)
    kf_right.predict()
    kf_right.update([signal_right[i]])
    mu_right.append(kf_right.x)
    cov_right.append(kf_right.P)

mu_front = np.array(mu_front)
cov_front = np.array(cov_front)
mu_left = np.array(mu_left)
cov_left = np.array(cov_left)
mu_right = np.array(mu_right)
cov_right = np.array(cov_right)

M_front,P_front,C_front = kf_front.rts_smoother(mu_front, cov_front)
M_left,P_left,C_left = kf_left.rts_smoother(mu_left, cov_left)
M_right,P_right,C_right = kf_right.rts_smoother(mu_right, cov_right)

# save data
fd = h5py.File('smoothed_sensor_reading.h5', 'w')
fd.create_dataset("time", data=time)
fd.create_dataset("sensor_front", data=M_front[:,0])
fd.create_dataset("sensor_left", data=M_left[:,0])
fd.create_dataset("sensor_right", data=M_right[:,0])
fd.create_dataset("att", data=fd_record['att_of_robot_0'][...])

# plot
fig, (ax0, ax1) = plt.subplots(nrows=2)
ax0.plot(signal_front, color = 'red')
ax0.plot(signal_left, color = 'yellow')
ax0.plot(signal_right, color = 'blue')
ax0.set_title('original sensor readings')

ax1.plot(M_front[:, 0], color = 'red')
ax1.plot(M_left[:,0], color = 'yellow')
ax1.plot(M_right[:,0], color = 'blue')
ax1.set_title('UKF-RTS smoothed readings')

plt.show()
