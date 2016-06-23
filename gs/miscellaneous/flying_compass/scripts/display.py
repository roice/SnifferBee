import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter

fd_foc = h5py.File('FOC_Record.h5', 'r+')
mox_reading = fd_foc['mox_reading'][...]
smoothed_mox_reading = fd_foc['smoothed_mox_reading'][...]

smoothed_mox_reading = smoothed_mox_reading[0:780]

normalized_mox_reading_f = (smoothed_mox_reading[:,0] - np.mean(smoothed_mox_reading[:,0]))/np.std(smoothed_mox_reading[:,0]);
normalized_mox_reading_l = (smoothed_mox_reading[:,1] - np.mean(smoothed_mox_reading[:,1]))/np.std(smoothed_mox_reading[:,1]);
normalized_mox_reading_r = (smoothed_mox_reading[:,2] - np.mean(smoothed_mox_reading[:,2]))/np.std(smoothed_mox_reading[:,2]);

blurred_mox_reading_f = gaussian_filter(normalized_mox_reading_f, sigma=20)
blurred_mox_reading_l = gaussian_filter(normalized_mox_reading_l, sigma=20)
blurred_mox_reading_r = gaussian_filter(normalized_mox_reading_r, sigma=20)

fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4)

ax0.plot(mox_reading[:,0], color = 'red')
ax0.plot(mox_reading[:,1], color = 'yellow')
ax0.plot(mox_reading[:,2], color = 'blue')
ax0.set_title('Original readings')

ax1.plot(smoothed_mox_reading[:,0], color = 'red')
ax1.plot(smoothed_mox_reading[:,1], color = 'yellow')
ax1.plot(smoothed_mox_reading[:,2], color = 'blue')
ax1.set_title('UKF filtered readings')

ax2.plot(normalized_mox_reading_f, color = 'red')
ax2.plot(normalized_mox_reading_l, color = 'green')
ax2.plot(normalized_mox_reading_r, color = 'blue')
ax2.set_title('Normalized filtered readings')

ax3.plot(blurred_mox_reading_f, color = 'red')
ax3.plot(blurred_mox_reading_l, color = 'green')
ax3.plot(blurred_mox_reading_r, color = 'blue')
ax3.set_title('blurred filtered readings')

plt.show()
