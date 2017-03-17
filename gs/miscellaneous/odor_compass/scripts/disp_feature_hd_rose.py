import h5py
import numpy as np
import matplotlib.pyplot as plt
import math

NUM_SENSORS = 3
MOX_DAQ_FREQ = 20
MOX_INTERP_FACTOR = 10
WT_LEVELS = 100
LEN_WAVELET = (3*MOX_DAQ_FREQ*MOX_INTERP_FACTOR)
LEN_RECENT_INFO = (15*MOX_DAQ_FREQ*MOX_INTERP_FACTOR)

fd = h5py.File('FOC_Record.h5', 'r+')
est_hd = fd['/FOC/est_direction'][...]

angle = []
for i in range(len(est_hd)):
    ang = math.atan2(-est_hd[i,0], est_hd[i,1])
    angle.append(ang)

theta = np.linspace(-np.pi, np.pi, 36)
r = np.zeros_like(theta)
for i in range(len(angle)):
    for j in range(len(theta)-1):
        if angle[i] > theta[j] and angle[i] < theta[j+1]:
            r[j] += 1.
            break
        else:
            continue

print r

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='polar')

#bars = ax.bar(theta[0:35], r[0:35], width=r[0:35]/len(angle), bottom=0.0)
bars = ax.bar(theta[0:35], r[0:35], width=0.1, bottom=0.0)

for radius, bar in zip(r[0:35], bars):
    bar.set_facecolor(plt.cm.jet(radius/10.))
    bar.set_alpha(0.75)

plt.show()
