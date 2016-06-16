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
t = h5py.File("smoothed_sensor_reading.h5", 'r+')['time'][...]
reading_front = h5py.File("smoothed_sensor_reading.h5", 'r+')['sensor_front'][...]
reading_left = h5py.File("smoothed_sensor_reading.h5", 'r+')['sensor_left'][...]
reading_right = h5py.File("smoothed_sensor_reading.h5", 'r+')['sensor_right'][...]

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

fig, (ax0, ax1, ax2) = plt.subplots(nrows=3)

ax0.plot(t, reading_front, 'red')
ax0.plot(t, reading_left, 'yellow')
ax0.plot(t, reading_right, 'blue')
ax0.set_title('smoothed sensor readings', fontdict=font);

ax1.plot(t, dr_front, color = 'red', linestyle = '-')
ax1.plot(t, dr_left, color = 'yellow', linestyle = '-')
ax1.plot(t, dr_right, color = 'blue', linestyle = '-')
ax1.set_title('1st order derivative', fontdict=font);

ax2.plot(t, ddr_front, color = 'red', linestyle = '-.')
ax2.plot(t, ddr_left, color = 'yellow', linestyle = '-.')
ax2.plot(t, ddr_right, color = 'blue', linestyle = '-.')
ax2.set_title('2nd order derivative', fontdict=font);

plt.show()
