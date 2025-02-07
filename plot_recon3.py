import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(2, 5, figsize=(25, 12))
data = pd.read_csv("vecs.csv").to_numpy()
axes = axes.flatten()

for idx, ax in enumerate(axes):
    row = data[idx]
    X = row[0:4200]
    h = row[4200:4700]
    X_hat = row[4700:]
    ax.plot(X, color="green", label="Original data")
    ax.plot(X_hat, color="blue", linestyle='--', dashes=(1, 3), label="Reconstructed data")
    ax.set_title(f"Sample {idx + 1}", fontsize=18)
    ax.set_xlabel("Band", fontsize=15)
    ax.set_ylabel("Reflectance (normalized)", fontsize=15)
    # if idx == 2:
    #     ax.legend(loc='upper center', ncol=2, fontsize=12, bbox_to_anchor=(0.5, 1.2))

lines, labels = axes[0].get_legend_handles_labels()
fig.legend(lines, labels, loc='upper center', ncol=2, fontsize=20, bbox_to_anchor=(0.5,1))

fig.subplots_adjust(top=0.9)
#plt.tight_layout()
plt.savefig("disc1.png")
#plt.show()

X_mean = np.mean(data[:,0:4200], axis=1)
X_hat_mean = np.mean(data[:,4700:], axis=1)


