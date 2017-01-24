import h5py
import numpy as np
import matplotlib.pyplot as plt
import math

DAQ_FREQ = 20
WIND_FREQ = 50

#fd = h5py.File('./Record_2017-01-05_10-12-29.h5', 'r+') // no wind
fd = h5py.File('./Record_2017-01-05_10-23-58.h5', 'r+')
#fd = h5py.File('./Record_2017-01-06_10-05-02.h5', 'r+')

att = fd['robot1/att'][...]
z3 = fd['robot1/debug/leso_z3'][...]
z3 = z3[WIND_FREQ*20:WIND_FREQ*3*60+WIND_FREQ*20]

yaw = []
for i in range(len(z3)):
    idx = int(i/float(WIND_FREQ)*float(DAQ_FREQ))
    if idx > len(att):
        break
    yaw.append(att[idx,2])
yaw = np.asarray(yaw)

offset = [-math.sin(310./360.*2.*np.pi)*0.12, math.cos(310./360.*2.*np.pi)*0.12]

angle = []
for i in range(len(z3)):
    temp_z3 = [z3[i,0]-offset[0], z3[i,1]-offset[1]]
    ang = math.atan2(-temp_z3[0], temp_z3[1])
    angle.append(ang)#+att[int(i/float(WIND_FREQ)*float(DAQ_FREQ)),2])
angle = np.asarray(angle)

t = np.asarray(range(len(angle)))/float(WIND_FREQ)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)

ax.plot(angle*180.0/np.pi, c='b');
ax.plot(yaw*180./np.pi, c='k')
ax.plot((angle+yaw)*180./np.pi, c='r')

plt.show()
