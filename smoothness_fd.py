import numpy as np
from ds_manager import DSManager

def get_mid_bands(size):
    x = np.linspace(0, 4199, size+2)
    x = x[1:-1]
    x = np.round(x)
    return x.astype(int)


ibs = {}

for key in [8, 16, 32, 64, 128, 256, 512]:
    fd_bands = get_mid_bands(key)
    ibs[key] = fd_bands

ds = DSManager(name="lucas",shuffle=False)
data = ds.data[:,0:-1]

fd_res = []

for key in [8, 16, 32, 64, 128, 256, 512]:
    fd_bands = ibs[key]
    data_initial = data[:, fd_bands]

    data_initial = ((data_initial[:, :-2] + data_initial[:, 1:-1] + data_initial[:, 2:]) / 3 - data_initial[:, 1:-1]) ** 2
    mean_derivative_initial = np.mean(np.sum(data_initial, axis=1))
    fd_res.append(mean_derivative_initial)

import matplotlib.pyplot as plt

plt.plot(fd_res)
plt.show()
fd_res = [float(i) for i in fd_res]
print(fd_res)
plt.show()

