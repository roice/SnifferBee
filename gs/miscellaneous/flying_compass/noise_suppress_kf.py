import h5py
import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
from scipy.linalg import block_diag
from filterpy.common import Q_discrete_white_noise

# get signal
signal_front = h5py.File('Record_2016-06-07_22-23-42.h5', 'r+')['sensors_of_robot_0'][...][:,0]
# inverse signal, high voltage correspond to high concentration
signal_front = 3.3 - signal_front

# Linear Kalman Filter
def sensor_lkf(alpha):
    lkf = KalmanFilter(dim_x=4, dim_z=1)
    dt = 0.1   # time step

    lkf.F = np.array([[0, -1./alpha, 1,  0],
                      [0,  1,        0,  0],
                      [0,  0,        1, dt],
                      [0,  0,        0,  1]])
    lkf.u = 0.
    lkf.H = np.array([[1, 0, 0, 0]])

    lkf.R = np.array([ [0.05**2] ])
    q = Q_discrete_white_noise(dim=2, dt=dt, var=0.01)
    q_y = np.array([[0,0],[0,0]])
    lkf.Q = block_diag(q, q_y)
    lkf.x = np.zeros(4)
    lkf.P = np.eye(4) * 500.
    return lkf

# filter running routine
def filter_data(kf, zs):
    xs, ps, lh = [], [], []
    for z in zs:
        kf.predict()
        kf.update(z)

        xs.append(kf.x)
        ps.append(kf.P.diagonal()) # just save variances
        lh.append(kf.likelihood)

    return np.asarray(xs), np.asarray(ps), np.asarray(lh)

# run filter
sensor_filter = sensor_lkf(0.5)
filtered_xs, ps, lh = filter_data(sensor_filter, signal_front)

# plot
fig, (ax0, ax1) = plt.subplots(nrows=2)
ax0.plot(signal_front, color = 'blue')
ax0.set_title('Voltage of front sensor')

ax0.plot(filtered_xs[:,0], color = 'red')

ax1.plot(lh)
ax1.set_title('Likelihood')

plt.show()
