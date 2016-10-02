import numpy as np
import math
import matplotlib.pyplot as plt

dt = 0.01   # s
m = 0.1     # kg
#noise = (np.random.rand(10000) - 0.5)*0.1
noise = np.zeros(10000)
#disturb = (np.random.rand(10000) - 0.5)*0.2+1.0  # N
#disturb = np.ones(10000)

t = np.asarray(range(10000))
disturb = np.sin(t*np.pi/100.0)*0.2+np.random.rand(10000)*0.1+1.0

temp_u = 0
temp_vel = 0
temp_y = 0
y = []
v = []
u = []
kp_pos = 1.0
kp_vel = 100.0
kd_vel = 100.0
ki_vel = 2.1
error = 0
d_error = 0
i_error = 0
for i in range(len(disturb)):
    temp_vel += (noise[i]+disturb[i]+temp_u)/m*dt
    temp_acc = -(noise[i]+disturb[i]+temp_u)/m*dt
    temp_y += temp_vel*dt
    dest_vel = (0 - temp_y)*kp_pos
    error = dest_vel - temp_vel
    d_error = -temp_acc
    i_error += error
    temp_u = (error*kp_vel + d_error*kd_vel + i_error*ki_vel)*dt # ref = 0
    y.append(temp_y)
    v.append(temp_vel)
    u.append(temp_u)

# leso
w0 = 10.
scale_u = 0.
temp_z1 = 0.
temp_z2 = 0.
temp_z3 = 0.
z1 = []
z2 = []
z3 = []
for i in range(len(y)):
    leso_err = y[i] - temp_z1
    temp_z1 += dt*(temp_z2 + 3*w0*leso_err)
    temp_z2 += dt*(temp_z3 + 3*pow(w0, 2)*leso_err + scale_u*u[i])
    temp_z3 += dt*(pow(w0, 3)*leso_err)
    z1.append(temp_z1)
    z2.append(temp_z2)
    z3.append(temp_z3)

fig, axes = plt.subplots(nrows=4, figsize=(10, 10))

axes[0].plot(y, color = 'red')
axes[0].plot(z1, color = 'blue')

axes[1].plot(v, color = 'red')
axes[1].plot(z2, color = 'blue')

axes[2].plot(disturb, color = 'red')
axes[2].plot(z3, color = 'blue')

axes[3].plot(u, color = 'green')

plt.show()
