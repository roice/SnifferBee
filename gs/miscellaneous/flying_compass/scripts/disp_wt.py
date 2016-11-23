import h5py
import numpy as np
import matplotlib.pyplot as plt

fd = h5py.File('FOC_Record.h5', 'r+')
wt_out = fd['/FOC/wt_out'][...]
wt_flag = fd['/FOC/wt_flag'][...]
wt_length = fd['/FOC/wt_length'][...]

print wt_flag

fig, axes = plt.subplots(nrows=len(wt_length)-1, ncols = 1, figsize=(6,6))
pointer = 0
for i in range(len(wt_length)-1):
    axes[i].plot(wt_out[pointer:pointer+wt_length[i,0]])
    pointer += wt_length[i,0]

plt.show()
