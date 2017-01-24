import h5py
import numpy as np
import matplotlib.pyplot as plt
import math

MOX_DAQ_FREQ = 20
START_UP_TIME = 20

#fd = h5py.File('Record_2017-01-20_12-34-37.h5', 'r+')
fd = h5py.File('Record_2017-01-20_17-06-56.h5', 'r+')
wind = fd['robot1/wind'][...]
wind = wind[START_UP_TIME*MOX_DAQ_FREQ:START_UP_TIME*MOX_DAQ_FREQ+2*60*MOX_DAQ_FREQ]

angle = []
for i in range(len(wind)):
    ang = math.atan2(-wind[i,0], wind[i,1])
    angle.append(ang)

strength = []
for i in range(len(wind)):
    strength.append(np.sqrt(wind[i,0]*wind[i,0]+wind[i,1]*wind[i,1]))

mean_wind = np.mean(wind, axis=0)
mean_wind_angle = math.atan2(-mean_wind[0], mean_wind[1])
mean_wind_strength = np.mean(np.asarray(strength))
std_wind_angle = np.std(np.asarray(angle)-mean_wind_angle)
std_wind_strength = np.std(np.asarray(strength))

print "mean wind = "+str(mean_wind)+", mean wind angle = "+str(mean_wind_angle*180./np.pi)+", mean wind strength = "+str(mean_wind_strength)+", std wind angle = "+str(std_wind_angle*180./np.pi)+", std wind strength = "+str(std_wind_strength)

'''
# Draw plot
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)
ax.plot(np.asarray(range(len(wind)))/float(MOX_DAQ_FREQ), wind[:,0], c='r')
ax.plot(np.asarray(range(len(wind)))/float(MOX_DAQ_FREQ), wind[:,1], c='b')
plt.show()
'''


# Draw original scatter polar
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='polar')
ax.scatter(angle, strength, s=1, color='k')
ax.scatter(math.atan2(-mean_wind[0], mean_wind[1]), np.sqrt(mean_wind[0]*mean_wind[0]+mean_wind[1]*mean_wind[1]), color='r')
plt.show()


'''
# Draw refined scatter polar
pos = fd['robot1/enu'][...]
pos_ref = np.asarray([0.2, -1.2, 1.4])
angle = []
strength = []
for i in range(len(wind)):
    err = pos[i] - pos_ref;
    if np.sqrt(err[0]*err[0]+err[1]*err[1]+err[2]*err[2]) > 0.5:
        continue
    ang = math.atan2(-wind[i,0], wind[i,1])
    angle.append(ang)
    strength.append(np.sqrt(wind[i,0]*wind[i,0]+wind[i,1]*wind[i,1]))

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='polar')
ax.scatter(angle, strength, s=1, color='k')
plt.show()
'''

'''
# Draw direction rose
theta = np.linspace(-np.pi, np.pi, 36)
r = np.zeros_like(theta)
for i in range(len(angle)):
    for j in range(len(theta)-1):
        if angle[i] > theta[j] and angle[i] < theta[j+1]:
            r[j] += 1.
            break
        else:
            continue

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='polar')

bars = ax.bar(theta[0:35], r[0:35], width=10.*6./360., bottom=0.0)

for radius, bar in zip(r[0:35], bars):
    bar.set_facecolor(plt.cm.jet(radius/10.))
    bar.set_alpha(0.75)

plt.show()
'''
