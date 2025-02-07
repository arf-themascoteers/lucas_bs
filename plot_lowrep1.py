import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from ds_manager import DSManager



fig, axes = plt.subplots(3, 1, figsize=(3, 7))
data = pd.read_csv("v2.csv").to_numpy()
ds = DSManager(name="min_lucas",shuffle=False)
data2 = ds.data[:,0:-1]

indices = [28, 126, 142, 164, 172, 198, 231, 295, 310, 321, 414, 501, 503, 517, 522, 525, 601, 612, 660, 661, 662, 664, 683, 700, 723, 760, 852, 934, 945, 974, 1017, 1024, 1032, 1035, 1050, 1057, 1065, 1090, 1129, 1135, 1220, 1230, 1254, 1265, 1316, 1425, 1526, 1556, 1597, 1599, 1641, 1764, 1850, 1872, 1890, 1921, 1922, 1924, 1976, 1981, 1983, 1984, 1987, 2015, 2084, 2138, 2142, 2144, 2148, 2149, 2187, 2252, 2275, 2368, 2373, 2513, 2532, 2533, 2536, 2540, 2596, 2654, 2756, 2783, 2900, 2918, 2930, 2959, 2989, 3001, 3018, 3047, 3153, 3183, 3203, 3275, 3318, 3319, 3470, 3475, 3477, 3478, 3568, 3608, 3610, 3733, 3736, 3741, 3766, 3775, 3820, 3899, 3903, 3933, 3967, 3969, 3985, 4115, 4144]
print(len(indices))
indices = [int(i) for i in indices]
mdata = data2[:,indices]

sindices = [1,3,9]

for ax in axes:
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_yticks(np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 3))

for i in range(3):
    ax = axes[i]
    sindex = sindices[i]
    row = data2[sindex]
    X = row
    h = data[i][4200:4700]
    h2 = mdata[sindex]


    ax.set_ylim([0, 1])


    ax.plot(X, color="green", label="Original data")
    ax.set_xlabel("Band",fontsize=14)
    ax.set_ylabel("Reflectance",fontsize=14)



plt.tight_layout()
plt.savefig(f"low_rep1.png")
plt.show()

