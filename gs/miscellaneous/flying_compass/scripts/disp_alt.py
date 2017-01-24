import h5py
import numpy as np
import matplotlib.pyplot as plt

fd = h5py.File('FOC_Record.h5', 'r+')
est = fd['/FOC/est_direction'][...]

theta = np.linspace(-np.pi, np.pi, 360)
r = np.zeros_like(theta)
for i in range(len(est)):
    for j in range(len(theta)-1):
        if est[i,2] > theta[j] and est[i,2] < theta[j+1]:
            r[j] += 1.
            break
        else:
            continue

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='polar')

ax.plot(theta, r)

plt.show()
