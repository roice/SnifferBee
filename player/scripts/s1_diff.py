import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from scipy import signal

fd = h5py.File('../data/Record_2016-07-08_15-24-37.h5', 'r+')
pos = fd['enu_of_robot_0'][...]
att = fd['att_of_robot_0'][...]

fd_foc = h5py.File('FOC_Record.h5', 'r+')
mox_reading = fd_foc['/FOC/mox_reading'][...]
mox_ukf_out = fd_foc['/FOC/mox_denoise'][...]

mox_ukf_out = fd['sensors_of_robot_0'][...]

# resample
mox_interp_f = signal.resample(mox_ukf_out[:,0], len(mox_ukf_out)*10)
mox_interp_l = signal.resample(mox_ukf_out[:,1], len(mox_ukf_out)*10)
mox_interp_r = signal.resample(mox_ukf_out[:,2], len(mox_ukf_out)*10)

# gaussian convolution
mox_smooth_f = gaussian_filter(mox_interp_f, sigma=100)
mox_smooth_l = gaussian_filter(mox_interp_l, sigma=100)
mox_smooth_r = gaussian_filter(mox_interp_r, sigma=100)

# derivative
mox_diff_f = np.diff(mox_smooth_f, n=2)
mox_diff_l = np.diff(mox_smooth_l, n=2)
mox_diff_r = np.diff(mox_smooth_r, n=2)

mox_diff_f /= np.std(mox_diff_f)
mox_diff_l /= np.std(mox_diff_l)
mox_diff_r /= np.std(mox_diff_r)

# save data
fd_diff = h5py.File('./mox_diff.h5', 'w')
grp = fd_diff.create_group("mox_diff")
dset_f = grp.create_dataset('front', (len(mox_diff_f),), dtype='f')
dset_l = grp.create_dataset('left', (len(mox_diff_l),), dtype='f')
dset_r = grp.create_dataset('right', (len(mox_diff_r),), dtype='f')
dset_f[...] = mox_diff_f;
dset_l[...] = mox_diff_l;
dset_r[...] = mox_diff_r;

fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4)

ax1.plot(mox_ukf_out[:,0], color = 'red')
ax1.plot(mox_ukf_out[:,1], color = 'yellow')
ax1.plot(mox_ukf_out[:,2], color = 'blue')
ax1.set_title('UKF filtered readings')

ax2.plot(mox_interp_f, color = 'red')
ax2.plot(mox_interp_l, color = 'yellow')
ax2.plot(mox_interp_r, color = 'blue')
ax2.set_title('Interpolated readings')

ax3.plot(mox_smooth_f, color = 'red')
ax3.plot(mox_smooth_l, color = 'yellow')
ax3.plot(mox_smooth_r, color = 'blue')
ax3.set_title('Gaussian convolution')

ax4.plot(mox_diff_f, color = 'red')
ax4.plot(mox_diff_l, color = 'yellow')
ax4.plot(mox_diff_r, color = 'blue')
ax4.set_title('2nd order derivative')


plt.show()
