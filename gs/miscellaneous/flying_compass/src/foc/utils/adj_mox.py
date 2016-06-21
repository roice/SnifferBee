import numpy as np
import matplotlib.pyplot as plt

alpha_r = 1;
alpha_d = 0.5;

t = np.linspace(0., 10, 100)

r = []
for i in range(len(t)):
    if len(r) is not 0:
        r_ = r[i-1]+alpha_r*(3.3-r[i-1])*0.1
        if r_ > 3.3:
            r_ = 3.3
        r.append(r_)
    else:
        r.append(0.8)

d = []
for i in range(len(t)):
    if len(d) is not 0:
        d_ = d[i-1]-alpha_d*(d[i-1]-0.8)*0.1
        if d_ < 0.8:
            d_ = 0.8
        d.append(d_)
    else:
        d.append(3.3)

fig, (ax0, ax1) = plt.subplots(nrows=2)
ax0.plot(t, r)
ax1.plot(t, d)
plt.show()
