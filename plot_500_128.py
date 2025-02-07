import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from ds_manager import DSManager

fig, axes = plt.subplots(3, 3, figsize=(10, 7))
#fig.subplots_adjust(hspace=0.5)
data = pd.read_csv("v2.csv").to_numpy()
ds = DSManager(name="min_lucas",shuffle=False)
data2 = ds.data[:,0:-1]

indices = [28, 126, 142, 164, 172, 198, 231, 295, 310, 321, 414, 501, 503, 517, 522, 525, 601, 612, 660, 661, 662, 664, 683, 700, 723, 760, 852, 934, 945, 974, 1017, 1024, 1032, 1035, 1050, 1057, 1065, 1090, 1129, 1135, 1220, 1230, 1254, 1265, 1316, 1425, 1526, 1556, 1597, 1599, 1641, 1764, 1850, 1872, 1890, 1921, 1922, 1924, 1976, 1981, 1983, 1984, 1987, 2015, 2084, 2138, 2142, 2144, 2148, 2149, 2187, 2252, 2275, 2368, 2373, 2513, 2532, 2533, 2536, 2540, 2596, 2654, 2756, 2783, 2900, 2918, 2930, 2959, 2989, 3001, 3018, 3047, 3153, 3183, 3203, 3275, 3318, 3319, 3470, 3475, 3477, 3478, 3568, 3608, 3610, 3733, 3736, 3741, 3766, 3775, 3820, 3899, 3903, 3933, 3967, 3969, 3985, 4115, 4144]
print(len(indices))
indices = [int(i) for i in indices]
mdata = data2[:,indices]

sindices = [1,3,9]

for i in range(3):
    sindex = sindices[i]
    row = data2[sindex]
    X = row
    h = data[i][4200:4700]
    h2 = mdata[sindex]

    axes[i, 0].set_ylim([0, 1])
    axes[i, 2].set_ylim([0, 1])

    axes[i,0].plot(X, color="green", label="Original data")
    #axes[i,0].set_title(f"Sample {i + 1}")
    axes[i,0].set_xlabel("Band",fontsize=14)
    axes[i,0].set_ylabel("Reflectance",fontsize=14)

    axes[i,1].plot(h, color="red", label="Compressed data")
    #axes[i,1].set_title(f"Compressed sample {i + 1} (SAE)")
    axes[i,1].set_xlabel("Index",fontsize=14)
    axes[i,1].set_ylabel("Value",fontsize=14)

    axes[i,2].plot(h2, color="blue", label="Compressed2 data")
    #axes[i,2].set_title(f"Compressed sample {i + 1} (AD)")
    axes[i,2].set_xlabel("Index",fontsize=14)
    axes[i,2].set_ylabel("Value",fontsize=14)


lines, labels = axes[0,1].get_legend_handles_labels()
#fig.legend(lines, labels, loc='upper center', ncol=2,bbox_to_anchor=(0.5,1))

#fig.subplots_adjust(top=0.9)
plt.tight_layout()
plt.savefig("low_rep.png")
plt.show()

