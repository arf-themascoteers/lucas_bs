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

    h = row[4200:4700]
    X_hat = row[4700:]
    #ax.set_ylim([0, 1])
    ax.plot(X, color="green", label="Original spectra")
    ax.plot(X_hat, color="red", linestyle='--', dashes=(1, 6), label="Reconstructed spectra")
    ax.set_title(f"Sample {idx + 1}", fontsize=16)
    ax.set_xlabel("Band", fontsize=14)
    ax.set_ylabel("Reflectance", fontsize=14)
    # if idx == 2:
    #     ax.legend(loc='upper center', ncol=2, fontsize=12, bbox_to_anchor=(0.5, 1.2))

lines, labels = axes[0].get_legend_handles_labels()
fig.legend(lines, labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5,1), fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.92])
#plt.tight_layout()
plt.savefig("recon.png",dpi=600)
plt.show()

X_mean = np.mean(data[:,0:4200], axis=1)
X_hat_mean = np.mean(data[:,4700:], axis=1)


