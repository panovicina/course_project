#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 3, 4096)
D = np.genfromtxt('pict.data')
plt.plot(x, D[0], label=r'$D_{1, 1}$')
plt.plot(x, D[1], label=r'$D_{1, 2}$')
plt.plot(x, D[2], label=r'$D_{1, 3}$')
plt.plot(x, D[3], label=r'$D_{2, 2}$')
plt.plot(x, D[4], label=r'$D_{2, 3}$')
plt.plot(x, D[5], label=r'$D_{3, 3}$')
plt.legend()
plt.show()
