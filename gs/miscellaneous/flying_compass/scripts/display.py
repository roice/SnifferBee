import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter

fd = h5py.File('../data/Record_2016-06-07_22-23-42.h5', 'r+')
pos = fd['enu_of_robot_0'][...]
att = fd['att_of_robot_0'][...]

fd_foc = h5py.File('FOC_Record.h5', 'r+')
mox_reading = fd_foc['mox_reading'][...]
mox_ukf_out = fd_foc['mox_ukf_output'][...]
mox_interp_out = fd_foc['mox_interp_output'][...]
mox_diff_out = fd_foc['mox_diff_output'][...]
mox_peak_time_f = fd_foc['mox_peak_time_0'][...]
mox_peak_time_l = fd_foc['mox_peak_time_1'][...]
mox_peak_time_r = fd_foc['mox_peak_time_2'][...]

mox_interp_out = mox_interp_out[0:1600]
mox_diff_out = mox_diff_out[0:900]

mox_ukf_out = mox_ukf_out[0:780]


mox_diff_out_f = gaussian_filter(mox_ukf_out[:,0], sigma = 3);
mox_diff_out_l = gaussian_filter(mox_ukf_out[:,1], sigma = 3);
mox_diff_out_r = gaussian_filter(mox_ukf_out[:,2], sigma = 3);

mox_diff_out_f = np.diff(mox_diff_out_f, n=2)
mox_diff_out_l = np.diff(mox_diff_out_l, n=2)
mox_diff_out_r = np.diff(mox_diff_out_r, n=2)

mox_diff_out_f /= np.std(mox_diff_out_f)
mox_diff_out_l /= np.std(mox_diff_out_l)
mox_diff_out_r /= np.std(mox_diff_out_r)

fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=5)

ax1.plot(mox_ukf_out[:,0], color = 'red')
ax1.plot(mox_ukf_out[:,1], color = 'yellow')
ax1.plot(mox_ukf_out[:,2], color = 'blue')
ax1.set_title('UKF filtered readings')

ax2.plot(mox_interp_out[:,0], color = 'red')
ax2.plot(mox_interp_out[:,1], color = 'yellow')
ax2.plot(mox_interp_out[:,2], color = 'blue')
ax2.set_title('Interpolated readings')

ax3.plot(mox_diff_out[:,0], color = 'red')
ax3.plot(mox_diff_out[:,1], color = 'yellow')
ax3.plot(mox_diff_out[:,2], color = 'blue')
ax3.set_title('1st Derivative readings')

ax4.scatter(mox_peak_time_f, np.ones_like(mox_peak_time_f), color = 'red')
ax4.scatter(mox_peak_time_l, np.ones_like(mox_peak_time_l)*2, color = 'yellow')
ax4.scatter(mox_peak_time_r, np.ones_like(mox_peak_time_r)*3, color = 'blue')
ax4.set_title('Peak readings')

ax5.plot(mox_diff_out_f, color = 'red')
ax5.plot(mox_diff_out_l, color = 'yellow')
ax5.plot(mox_diff_out_r, color = 'blue')
ax5.set_title('1st Derivative readings')

plt.show()
