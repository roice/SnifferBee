import h5py
import numpy as np
import matplotlib.pyplot as plt
import math

FREQ = 50

#fd = h5py.File('./Record_2017-01-05_10-23-58.h5', 'r+')
fd = h5py.File('./Record_2017-01-05_10-12-29.h5', 'r+')
z3 = fd['robot1/debug/leso_z3'][...]
z3 = z3[FREQ*20:FREQ*3*60+FREQ*20]

offset = [-math.sin(310./360.*2.*np.pi)*0.12, math.cos(310./360.*2.*np.pi)*0.12]

angle = []
r = []
for i in range(len(z3)):
    temp_z3 = [z3[i,0]-offset[0], z3[i,1]-offset[1]]
    ang = math.atan2(-temp_z3[0], temp_z3[1])
    radius = math.sqrt(temp_z3[0]*temp_z3[0]+temp_z3[1]*temp_z3[1])
    angle.append(ang)
    r.append(radius)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='polar')

ax.scatter(angle, r, s = 1, cmap=plt.cm.hsv);

plt.show()
