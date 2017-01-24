import h5py
import numpy as np
import matplotlib.pyplot as plt
import math

ANEMO_FREQ = 50

#fd = h5py.File('Record_2017-01-20_11-52-54.h5', 'r+')
#fd = h5py.File('Record_2017-01-20_16-44-05.h5', 'r+')
fd = h5py.File('Record_2017-01-20_17-54-03.h5', 'r+')
wind = fd['anemometers/1'][...]

angle = []
for i in range(len(wind)):
    ang = math.atan2(-wind[i,0], wind[i,1])
    angle.append(ang)

strength = []
for i in range(len(wind)):
    strength.append(np.sqrt(wind[i,0]*wind[i,0]+wind[i,1]*wind[i,1]))

mean_wind = np.mean(wind, axis=0)
mean_wind_angle = np.mean(np.asarray(angle))
mean_wind_strength = np.mean(np.asarray(strength))
std_wind_angle = np.std(np.asarray(angle))
std_wind_strength = np.std(np.asarray(strength))

print "mean wind = "+str(mean_wind)+", mean wind angle = "+str(mean_wind_angle*180./np.pi)+", mean wind strength = "+str(mean_wind_strength)+", std wind angle = "+str(std_wind_angle*180./np.pi)+", std wind strength = "+str(std_wind_strength)

'''
# Draw plot
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)
ax.plot(np.asarray(range(len(wind)))/float(ANEMO_FREQ), wind[:,0], c='r')
ax.plot(np.asarray(range(len(wind)))/float(ANEMO_FREQ), wind[:,1], c='b')
plt.show()
'''

# Draw direction
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)
ax.plot(np.asarray(range(len(angle)))/float(ANEMO_FREQ), np.asarray(angle)*180./np.pi, c='r')
#ax.plot(np.asarray(range(len()))/float(ANEMO_FREQ), wind[:,1], c='b')
plt.show()

'''
# Draw scatter
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='polar')
ax.scatter(angle, strength, s=1, color='k')
ax.scatter(math.atan2(-mean_wind[0], mean_wind[1]), np.sqrt(mean_wind[0]*mean_wind[0]+mean_wind[1]*mean_wind[1]), color='r')
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
