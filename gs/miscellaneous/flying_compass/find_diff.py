import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.signal import correlate

'''
# Example
a = 0
b = 10
N = 10000
t = np.linspace(0, 10, N)
reading_front = (np.sin(t/10.*2*np.pi)+1.)/2.
reading_front -= np.mean(reading_front)
reading_left = (np.sin((t+0.2)/10.*2*np.pi)+1.)/2.
reading_left -= np.mean(reading_left)
reading_right = (np.sin((t+0.6)/10.*2*np.pi)+1.)/2.
reading_right -= np.mean(reading_right)
time = np.arange(1-N, N) # centered at 0
'''

font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }

# get sensor reading
fd_r = h5py.File("smoothed_sensor_reading.h5", 'r+')
t = fd_r['time'][...]
reading_front = fd_r['sensor_front'][...]
reading_left = fd_r['sensor_left'][...]
reading_right = fd_r['sensor_right'][...]
att = fd_r['att'][...]

# preprocess
reading_front -= np.mean(reading_front)
reading_left -= np.mean(reading_left)
reading_right -= np.mean(reading_right)
reading_front /= np.std(reading_front)
reading_left /= np.std(reading_left)
reading_right /= np.std(reading_right)

'''
# Method 1: overall correlation
time = np.arange(1-len(t), len(t)) * 0.1
cr_front_left = correlate(reading_front, reading_left)
shift_front_left = time[cr_front_left.argmax()]
cr_front_right = correlate(reading_front, reading_right)
shift_front_right = time[cr_front_right.argmax()]
cr_left_right = correlate(reading_left, reading_right)
shift_left_right = time[cr_left_right.argmax()]

print("correlation front & left is "+str(shift_front_left))
print("correlation front & rightt is "+str(shift_front_right))
print("correlation left & right is "+str(shift_left_right))
'''

# Method 2: interval-by-interval
shift_fl, shift_fr, shift_lr = [], [], []
for idx in range(len(t)):
    if idx%10 == 0:
        front, left, right = [], [], []
    front.append(reading_front[idx])
    left.append(reading_left[idx])
    right.append(reading_right[idx])
    if idx%10 == 9:
        time = np.arange(1-10, 10)*0.1
        shift_fl.append(time[correlate(np.array(front), np.array(left)).argmax()])
        shift_fr.append(time[correlate(np.array(front), np.array(right)).argmax()])
        shift_lr.append(time[correlate(np.array(left), np.array(right)).argmax()])

direction = []
for idx in range(len(shift_fl)):
    f = 0
    l = f - shift_fl[idx]
    r = f - shift_fr[idx]
    if l == 0 or r == 0:
        if len(direction) != 0:
            direction.append(direction[-1])
        else:
            direction.append(0)
        continue
    flr = np.array([f,l,r])
    index = np.array([0,1,2])
    first_sensor_idx = flr.argmin()
    second_sensor_idx = np.where(flr == flr[flr != flr.min()].min())[0][0]
    third_sensor_idx = index[(index != first_sensor_idx)&(index != second_sensor_idx)][0]
    base_angle = 120*first_sensor_idx
    if second_sensor_idx == first_sensor_idx + 1 \
            or second_sensor_idx == 0 and first_sensor_idx == 2:
        sign = 1.
    else:
        sign = -1.
    t1 = abs(flr[third_sensor_idx]-flr[second_sensor_idx])
    t2 = abs(flr[first_sensor_idx]-flr[second_sensor_idx])
    angle = np.arctan(1.732*t1/(t1+2*t2))*180./np.pi
    dir_angle = base_angle + sign*angle
    direction.append(dir_angle)

fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4)

ax0.plot(t, reading_front, 'red')
ax0.plot(t, reading_left, 'yellow')
ax0.plot(t, reading_right, 'blue')
ax0.set_title('smoothed sensor readings', fontdict=font);

ax1.plot(shift_fl)
ax1.plot(shift_fr)
ax1.plot(shift_lr)

ax2.plot(direction)

direction = np.array(direction)
for i in range(len(direction)):
    direction[i] += att[:,2][i*10]*180./np.pi

ax3.plot(direction)

plt.show()
