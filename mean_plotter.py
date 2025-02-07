import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("mean.csv").to_numpy()
data = data.reshape(-1)

res = pd.read_csv("results/16/bsdrfc_lucas_16_robust_dyn_0.csv").to_numpy()
res = res[0:25,-16:]
indices = [0,12,24]
res = res[indices]
res = res.astype(int)

titles = ["Initial state","Midpoint of training","End of training"]

fig, axes = plt.subplots(3, 1, figsize=(10, 15))

for i, ax in enumerate(axes):
    x_points = res[i]
    y_points = data[x_points]
    colors = plt.cm.plasma(np.linspace(0, 1, len(x_points)))
    axes[i].plot(data)
    for j, (x, y) in enumerate(zip(x_points, y_points)):
        axes[i].scatter(x, y, color=colors[j], s=50, zorder=5)
        offset = (-20 if j % 2 == 0 else 20)
        y_offset = (-0.02 if j % 2 == 0 else 0.02)
        axes[i].text(x, y+y_offset, f"{x}", fontsize=10, ha="center", va="bottom", color="black")
    axes[i].set_title(titles[i], fontsize=25)
    axes[i].set_ylim([0, 0.5])
    axes[i].set_xlabel("Band", fontsize=20)
    axes[i].set_ylabel("Reflectance", fontsize=20)
    axes[i].tick_params(axis='x', labelsize=12)
    axes[i].tick_params(axis='y', labelsize=12)

plt.tight_layout()
plt.savefig("bandc.png",dpi=600)
plt.show()

