import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from ds_manager import DSManager


ds = DSManager(name="min_lucas")
fig, axes = plt.subplots(3, 1, figsize=(5, 7))
data = pd.read_csv("vecs3.csv").to_numpy()
data[:, 0:4200] = ds.scaler_X.inverse_transform(data[:, 0:4200])
data[:, 4700:] = ds.scaler_X.inverse_transform(data[:, 4700:])
axes = axes.flatten()
xinds = [3,5,6]

for idx, ax in enumerate(axes):
    ax.tick_params(axis='both', which='major', labelsize=12)
    #ax.set_yticks(np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 3))

    xins = xinds[idx]
    row = data[xins]
    X = row[0:4200]

    indices = [28, 126, 142, 164, 172, 198, 231, 295, 310, 321, 414, 501, 503, 517, 522, 525, 601, 612, 660, 661, 662,
               664, 683, 700, 723, 760, 852, 934, 945, 974, 1017, 1024, 1032, 1035, 1050, 1057, 1065, 1090, 1129, 1135,
               1220, 1230, 1254, 1265, 1316, 1425, 1526, 1556, 1597, 1599, 1641, 1764, 1850, 1872, 1890, 1921, 1922,
               1924, 1976, 1981, 1983, 1984, 1987, 2015, 2084, 2138, 2142, 2144, 2148, 2149, 2187, 2252, 2275, 2368,
               2373, 2513, 2532, 2533, 2536, 2540, 2596, 2654, 2756, 2783, 2900, 2918, 2930, 2959, 2989, 3001, 3018,
               3047, 3153, 3183, 3203, 3275, 3318, 3319, 3470, 3475, 3477, 3478, 3568, 3608, 3610, 3733, 3736, 3741,
               3766, 3775, 3820, 3899, 3903, 3933, 3967, 3969, 3985, 4115, 4144]

    print(len(indices))

    X_hat = X[indices]
    ax.plot(X, color="green", label="Original spectra")
    ax.plot(indices, X_hat, color="blue", linestyle='--', dashes=(1, 2), label="Reconstructed spectra", linewidth=3)
    ax.set_title(f"Sample {idx + 1}", fontsize=16)
    ax.set_xlabel("Band", fontsize=14)
    ax.set_ylabel("Reflectance", fontsize=14)
    # if idx == 2:
    #     ax.legend(loc='upper center', ncol=2, fontsize=12, bbox_to_anchor=(0.5, 1.2))

lines, labels = axes[0].get_legend_handles_labels()
fig.legend(lines, labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5,1), fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.92])
#plt.tight_layout()
plt.savefig("reconp.png",dpi=600)
plt.show()

X_mean = np.mean(data[:,0:4200], axis=1)
X_hat_mean = np.mean(data[:,4700:], axis=1)


