import h5py
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter

fd = h5py.File("Record_2016-09-08_22-49-15.h5", 'r+')
vel_p = fd['/robot1/vel_p'][...]
z3 = fd['/robot1/leso_z3'][...]
wind_p = fd['/robot1/wind'][...]
anemo_1 = fd['/robot1/anemometer_1'][...]
anemo_2 = fd['/robot1/anemometer_2'][...]
anemo_3 = fd['/robot1/anemometer_3'][...]

alpha = 0.007

vel_p_0 = gaussian_filter(vel_p[:,0], sigma = 5)
vel_p_1 = gaussian_filter(vel_p[:,1], sigma = 5)
vel_p_2 = gaussian_filter(vel_p[:,2], sigma = 5)
#vel_p = np.asarray([ vel_p_0, vel_p_1, vel_p_2 ])

anemo = []
for i in range(len(anemo_1)):
    temp_anemo = [ (anemo_1[i,0]+anemo_2[i,0]+anemo_3[i,0])/3.0, (anemo_1[i,1]+anemo_2[i,1]+anemo_3[i,1])/3.0, (anemo_1[i,2]+anemo_2[i,2]+anemo_3[i,2])/3.0 ]
    anemo.append(temp_anemo)
anemo = np.asarray(anemo)

wind_p = []
for i in range(len(z3)):
    temp_wind_p = [ vel_p_0[i]-alpha*z3[i,0], vel_p_1[i]-alpha*z3[i,1], vel_p_2[i]-alpha*z3[i,2] ]
    wind_p.append(temp_wind_p)
wind_p = np.asarray(wind_p)

wind_direct = []
for i in range(len(wind_p)):
    wind_direct.append(math.atan2(-wind_p[i,0], wind_p[i,1]))
wind_direct = np.asarray(wind_direct)

print("std vel_p_0 is "+str(np.std(vel_p_0)))
print("std vel_p_1 is "+str(np.std(vel_p_1)))
print("std z3[:,0] is "+str(np.std(z3[:,0])))
print("std z3[:,1] is "+str(np.std(z3[:,1])))

fig, axes = plt.subplots(nrows=5, figsize=(6,6))

#axes[0].plot(vel_p[:,0], color = 'blue')
#axes[0].plot(vel_p[:,1], color = 'green')
axes[0].plot(vel_p_0, color = 'blue')
axes[0].plot(vel_p_1, color = 'green')


axes[1].plot(alpha*z3[:,0], color = 'blue')
axes[1].plot(alpha*z3[:,1], color = 'green')

axes[2].plot(anemo)

axes[3].plot(wind_p)

axes[4].plot(wind_direct)

plt.show()
