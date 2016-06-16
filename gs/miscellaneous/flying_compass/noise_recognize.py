import h5py
import numpy as np
import matplotlib.pyplot as plt

# get noise sequence
noise = h5py.File('Record_2016-06-08_10-54-32.h5', 'r+')['sensors_of_robot_0'][...]
time = h5py.File('Record_2016-06-08_10-54-32.h5', 'r+')['time_of_robot_0'][...]

dt = 0.1 # 0.1 s
N = len(time)

# DFT
ft_noise_front = np.fft.fft(noise[:, 0])

f = np.linspace(0., 1.0/(2.0*dt), N/2)

# standard deviation
print("standard deviation of front is " + str(np.std(noise[:,0])))
print("standard deviation of left is " + str(np.std(noise[:,1])))
print("standard deviation of right is " + str(np.std(noise[:,2])))

fig, (ax0, ax1) = plt.subplots(nrows=2)
ax0.plot(noise)
ax0.set_title('Noise of sensor readings')

ax1.plot(np.abs(ft_noise_front))

plt.show()
