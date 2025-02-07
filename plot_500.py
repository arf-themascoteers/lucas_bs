import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from ds_manager import DSManager

fig, axes = plt.subplots(3, 2, figsize=(7, 7))
#fig.subplots_adjust(hspace=0.5)
data = pd.read_csv("v2.csv").to_numpy()
ds = DSManager(name="min_lucas",shuffle=False)
data2 = ds.data[:,0:-1]



for i in range(3):
    row = data[i]
    X = data2[i]
    h = row[4200:4700]

    axes[i, 0].set_ylim([0, 1])

    axes[i,0].plot(X, color="green", label="Original data")
    axes[i,0].set_title(f"Sample {i + 1}")
    axes[i,0].set_xlabel("Band")
    axes[i,0].set_ylabel("Reflectance")

    axes[i,1].plot(h, color="red", label="Compressed data")
    axes[i,1].set_title(f"Latent vector for sample {i + 1}")
    axes[i,1].set_xlabel("Index")
    axes[i,1].set_ylabel("Value")


lines, labels = axes[0,1].get_legend_handles_labels()
#fig.legend(lines, labels, loc='upper center', ncol=2,bbox_to_anchor=(0.5,1))

#fig.subplots_adjust(top=0.9)
plt.tight_layout()
plt.savefig("low_rep_sae.png")
plt.show()

