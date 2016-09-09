import numpy as np
import matplotlib.pyplot as plt

dt = 0.01   # s
m = 1.0     # kg
u = np.random.rand(10000)  # N
temp_y = 0.0
temp_vel = 0.0
y = []
for i in range(len(u)):
    temp_vel += u[i]/m*dt
    temp_y += temp_vel*dt
    y.append(temp_y)

# leso
w0 = 10.0
temp_z1 = 0.
temp_z2 = 0.
temp_z3 = 0.
z1 = []
z2 = []
z3 = []
for i in range(len(u)):
    leso_err = y[i] - temp_z1
    temp_z1 += dt*(temp_z2 + 3*w0*leso_err)
    temp_z2 += dt*(temp_z3 + 3*pow(w0, 2)*leso_err + u[i])
    temp_z3 += dt*(pow(w0, 3)*leso_err)
    z1.append(temp_z1)
    z2.append(temp_z2)
    z3.append(temp_z3)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)

ax.plot(y, color = 'blue')
ax.plot(z1, color = 'green')
ax.plot(z3, color = 'red')

plt.show()
