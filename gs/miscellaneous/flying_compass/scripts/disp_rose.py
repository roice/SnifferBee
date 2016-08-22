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
direction = fd['/FOC/direction']
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

# for line plotting
n = 100
theta = np.asarray([i*2.*np.pi/n for i in range(100)])
r = np.zeros_like(theta)
for i in range(len(direction)):
    ang = math.atan2(-direction[i,0], direction[i,1]) + np.pi
    for j in range(len(theta)):
        if ang < theta[j]:
            r[j] += 1
            break

sum_theta = 0
sum_r = 0
for i in range(len(theta)):
    sum_theta += theta[i]*r[i]
    sum_r += r[i]
main_theta = sum_theta/sum_r
main_r = sum_r/len(theta)


main_theta = 0
for i in range(len(direction)):
    ang = math.atan2(direction[i,1], direction[i,0])
    main_theta += ang
main_theta /= len(direction)
main_r = 100


fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='polar')

'''
ax.scatter(theta, r, c=colors, s=area, cmap=plt.cm.hsv)
ax.set_alpha(0.75)
'''

ax.plot(theta, r)

ax.scatter(main_theta, main_r, s = 100)

plt.show()
