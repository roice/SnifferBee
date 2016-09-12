import numpy as np
import matplotlib.pyplot as plt

dt = 0.01   # s
m = 1.0     # kg
noise = (np.random.rand(10000) - 0.5)*0.2
disturb = np.random.rand(10000) - 0.5  # N
#disturb = np.ones(10000)

temp_u = 0
temp_vel = 0
temp_y = 0
y = []
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
    u.append(temp_u)

# leso
w0 = 100.0
temp_z1 = 0.
temp_z2 = 0.
temp_z3 = 0.
z1 = []
z2 = []
z3 = []
for i in range(len(y)):
    leso_err = y[i] - temp_z1
    temp_z1 += dt*(temp_z2 + 3*w0*leso_err)
    temp_z2 += dt*(temp_z3 + 3*pow(w0, 2)*leso_err + u[i])
    temp_z3 += dt*(pow(w0, 3)*leso_err)
    z1.append(temp_z1)
    z2.append(temp_z2)
    z3.append(temp_z3)

fig, axes = plt.subplots(nrows=3, figsize=(10, 10))

axes[0].plot(y, color = 'blue')
axes[0].plot(z1, color = 'green')

axes[1].plot(disturb, color = 'red')
axes[1].plot(z3, color = 'green')

axes[2].plot(u, color = 'blue')

plt.show()
