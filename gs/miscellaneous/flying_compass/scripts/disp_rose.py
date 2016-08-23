import h5py
import numpy as np
import matplotlib.pyplot as plt
import math

fd = h5py.File('FOC_Record.h5', 'r+')
#grad = fd['/FOC/mox_gradient'][...]
#edge_max = fd['/FOC/mox_edge_max'][...]
#edge_min = fd['/FOC/mox_edge_min'][...]
#cp_max = fd['/FOC/mox_cp_max'][...]
#cp_min = fd['/FOC/mox_cp_min'][...]
direction = fd['/FOC/est_direction']
belief = fd['/FOC/est_belief']
dt = fd['/FOC/est_dt']
#wind = fd['/FOC/wind']
#att = h5py.File('../data/Record_2016-08-03_17-30-06.h5', 'r+')['robot1/att'][...]

'''
# for scatter plotting
theta = []
for i in range(len(direction)):
    ang = math.atan2(direction[i,1], direction[i,0])
    theta.append(ang)
theta = np.asarray(theta)
r = np.ones_like(theta)
area = 20*np.ones_like(r)
colors = theta
'''
print len(direction)

'''
# for line plotting
n = 100
theta = np.asarray([(i-n)*np.pi/n for i in range(2*n)])
r = np.zeros_like(theta)
for i in range(len(direction)):
    if dt[i] < 20:
        continue;
    ang = math.atan2(-direction[i,0], direction[i,1])
    for j in range(len(theta)):
        if ang < theta[j]:
            r[j] += 1
            break
'''


theta = []
r_belief = []
r_dt = []
for i in range(len(direction)):
    if dt[i] < 10:
        continue
    ang = math.atan2(-direction[i,0], direction[i,1])
    theta.append(ang)
    r_belief.append(belief[i])
    r_dt.append(dt[i])
r_belief = np.asarray(r_belief)
r_belief /= np.std(r_belief)
r_dt = np.asarray(r_dt)
#r_dt /= np.std(r_dt)

sum_d_x = 0
sum_d_y = 0
for i in range(len(direction)):
    if (dt[i]) < 10:
        continue
    sum_d_x += direction[i,0]#*belief[i]
    sum_d_y += direction[i,1]#*belief[i]
main_theta = math.atan2(-sum_d_x, sum_d_y)
main_r = 100



fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='polar')


ax.scatter(theta, r_dt, s=1, cmap=plt.cm.hsv)
ax.set_alpha(0.75)



#ax.plot(theta, r)

ax.scatter(main_theta, main_r, s = 100)

plt.show()
