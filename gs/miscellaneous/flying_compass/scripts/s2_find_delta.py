import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from scipy import signal


fd = h5py.File('./mox_diff.h5', 'r+')
mox_diff_f = fd['mox_diff/front']
mox_diff_l = fd['mox_diff/left']
mox_diff_r = fd['mox_diff/right']

# calculate delta t
win_len = 5 # 5 second window
shift_fl, shift_fr, shift_lr = [], [], []
corr_max_fl, corr_max_fr, corr_max_lr = [], [], []
for i in range(len(mox_diff_f)/10): # update at 10 Hz
    if i < win_len*10:
        continue
    else:
        front, left, right = [], [], []
        for idx in range((i-win_len*10)*10, i*10):
            front.append(mox_diff_f[idx])
            left.append(mox_diff_l[idx])
            right.append(mox_diff_r[idx])
        time = np.arange(1-win_len*100, win_len*100)*0.01 # -(N-1)~(N-1)
        corr_fl = signal.correlate(np.array(front), np.array(left))
        corr_fr = signal.correlate(np.array(front), np.array(right))
        corr_lr = signal.correlate(np.array(left), np.array(right))
        corr_max_fl.append(corr_fl.max())
        corr_max_fr.append(corr_fr.max())
        corr_max_lr.append(corr_lr.max())
        shift_fl.append(time[corr_fl.argmax()])
        shift_fr.append(time[corr_fr.argmax()])
        shift_lr.append(time[corr_lr.argmax()])

# save data
fd_out = h5py.File('./correlation.h5', 'w')
grp = fd_out.create_group("corr")
dset_max_fl = grp.create_dataset('corr_max_fl', (len(corr_max_fl),), dtype='f')
dset_max_fr = grp.create_dataset('corr_max_fr', (len(corr_max_fr),), dtype='f')
dset_max_lr = grp.create_dataset('corr_max_lr', (len(corr_max_lr),), dtype='f')
dset_shift_fl = grp.create_dataset('corr_shift_fl', (len(shift_fl),), dtype='f')
dset_shift_fr = grp.create_dataset('corr_shift_fr', (len(shift_fr),), dtype='f')
dset_shift_lr = grp.create_dataset('corr_shift_lr', (len(shift_lr),), dtype='f')
dset_max_fl[...] = corr_max_fl
dset_max_fr[...] = corr_max_fr
dset_max_lr[...] = corr_max_lr
dset_shift_fl[...] = shift_fl
dset_shift_fr[...] = shift_fr
dset_shift_lr[...] = shift_lr

fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4)

ax1.plot(mox_diff_f, color = 'red')
ax1.plot(mox_diff_l, color = 'yellow')
ax1.plot(mox_diff_r, color = 'blue')
ax1.set_title('MOX diff')

ax2.plot(shift_fl, color = 'red')
ax2.plot(shift_fr, color = 'yellow')
ax2.plot(shift_lr, color = 'blue')
ax2.set_title('Time shift')

ax3.plot(corr_max_fl, color = 'red')
ax3.plot(corr_max_fr, color = 'yellow')
ax3.plot(corr_max_lr, color = 'blue')
ax3.set_title('Corr max')

plt.show()
