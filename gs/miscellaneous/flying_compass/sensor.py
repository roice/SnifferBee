import numpy as np
import matplotlib.pyplot as plt
import h5py

font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }

def pulse(t, t_r, t_f):
    if t < t_r or t > t_f:
        return 0
    else:
        return 1

# c(k+1) - c(k) = dt * (- alpha * c(k) + alpha * conc(k+1))
class Sensor:
    c = 0
    t = 0
    baseline = 0
    def __init__(self, baseline, alpha, tau):
        self.baseline = baseline
        self.alpha = alpha
        self.tau = tau

    def update(self, conc, time):
        dt = time - self.t
        self.t = time
        self.c = self.c + dt*(-self.alpha*self.c+self.alpha*conc)
        if self.c > self.tau:
            return self.c + self.baseline
        else:
            return 0 + self.baseline

sensor_front = Sensor(0.8, 0.9, 0)
sensor_left = Sensor(1, 1.2, 0)
sensor_right = Sensor(1.5, 1.0, 0)

t = np.arange(0, 20, 0.1) # 10 Hz, 20 s

real_conc_front = np.array([ 3*pulse(i, 1.0, 6) for i in t ])
real_conc_left = np.array([ 3*pulse(i, 1.2, 6.2) for i in t ])
real_conc_right = np.array([ 3*pulse(i, 1.3, 6.3) for i in t ])

reading_front = np.array([ sensor_front.update(real_conc_front[i], t[i]) for i in range(0, len(t)) ])
reading_left = np.array([ sensor_left.update(real_conc_left[i], t[i]) for i in range(0, len(t)) ])
reading_right = np.array([ sensor_right.update(real_conc_right[i], t[i]) for i in range(0, len(t)) ])

# save data
fd = h5py.File('sensor_reading.h5', 'w')
fd.create_dataset("time", data=t)
fd.create_dataset("sensor_front", data=reading_front)
fd.create_dataset("sensor_left", data=reading_left)
fd.create_dataset("sensor_right", data=reading_right)

#plt.plot(t, real_conc, 'k')
plt.plot(t, reading_front, 'red')
plt.plot(t, reading_left, 'yellow')
plt.plot(t, reading_right, 'blue')
plt.title('Gas sensor reading', fontdict=font)
plt.xlabel('time (s)', fontdict=font)
plt.ylabel('voltage (V)', fontdict=font)
plt.ylim(0, 4)
plt.show()
