import numpy as np
import matplotlib.pyplot as plt
import h5py
import sympy

font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }

def derivative_1st_order(x_pre, x_next, dt):
    return (x_next-x_pre)/dt;


# get sensor reading
t = h5py.File("sensor_reading.h5", 'r+')['time'][...]
reading_front = h5py.File("sensor_reading.h5", 'r+')['sensor_front'][...]
reading_left = h5py.File("sensor_reading.h5", 'r+')['sensor_left'][...]
reading_right = h5py.File("sensor_reading.h5", 'r+')['sensor_right'][...]

# 1st-order derivative
dr_front = np.array([0] + [ derivative_1st_order(reading_front[i], reading_front[i+1], t[i+1]-t[i]) for i in range(len(t)-1) ])
dr_left = np.array([0] + [ derivative_1st_order(reading_left[i], reading_left[i+1], t[i+1]-t[i]) for i in range(len(t)-1) ])
dr_right = np.array([0] + [ derivative_1st_order(reading_right[i], reading_right[i+1], t[i+1]-t[i]) for i in range(len(t)-1) ])

# fetch positive of 1st-order derivative, i.e., rising of gas sensor
for i in range(len(t)):
    if dr_front[i] < 0:
        dr_front[i] = 0
    if dr_left[i] < 0:
        dr_left[i] = 0
    if dr_right[i] < 0:
        dr_right[i] = 0

# 2nd-order derivative
ddr_front = np.array([0] + [ derivative_1st_order(dr_front[i], dr_front[i+1], t[i+1]-t[i]) for i in range(len(t)-1) ])
ddr_left = np.array([0] + [ derivative_1st_order(dr_left[i], dr_left[i+1], t[i+1]-t[i]) for i in range(len(t)-1) ])
ddr_right = np.array([0] + [ derivative_1st_order(dr_right[i], dr_right[i+1], t[i+1]-t[i]) for i in range(len(t)-1) ])

cr_front_left = np.correlate(dr_front, dr_left, 'full')
#cr_front_left = np.correlate(reading_front, reading_left, 'full')

print "front and left: " + str(np.nonzero(cr_front_left == cr_front_left.max())[0])

plt.plot(t, reading_front, 'red')
plt.plot(t, reading_left, 'yellow')
plt.plot(t, reading_right, 'blue')
plt.plot(t, dr_front, color = 'red', linestyle = '--')
plt.plot(t, dr_left, color = 'yellow', linestyle = '--')
plt.plot(t, dr_right, color = 'blue', linestyle = '--')
plt.plot(t, ddr_front, color = 'red', linestyle = '-.')
plt.plot(t, ddr_left, color = 'yellow', linestyle = '-.')
plt.plot(t, ddr_right, color = 'blue', linestyle = '-.')
plt.title('Gas sensor reading', fontdict=font)
plt.xlabel('time (s)', fontdict=font)
plt.ylabel('voltage (V)', fontdict=font)
plt.ylim(0, 4)
plt.show()
