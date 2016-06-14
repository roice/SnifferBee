import h5py
import matplotlib.pyplot as plt

enu_1 = h5py.File('Record_2016-06-08_10-54-32.h5', 'r+')['sensors_of_robot_0'][...]



plt.plot(enu_1)


plt.show()
