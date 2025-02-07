import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("mean.csv").to_numpy()
data = data.reshape(-1)

res = pd.read_csv("results/16/bsdrfc_lucas_16_robust_dyn_0.csv").to_numpy()
res = res[0:25, -16:]
indices = [0, 12, 24]
res = res[indices]
res = res.astype(int)

titles = ["Initial state", "Midpoint of training", "End of training"]
labels = ["(a)", "(b)", "(c)"]

fig, axes = plt.subplots(3, 1, figsize=(12, 15))

def get_y_offset(j):
    if j%6 == 0:
        return 0.12
    if j%5 == 0:
        return -0.12
    if j%4 == 0:
        return 0.04
    if j%3 == 0:
        return -0.04
    if j%2 == 0:
        return 0.08
    return -0.08

for i, ax in enumerate(axes):
    x_points = res[i]
    y_points = data[x_points]
    colors = plt.cm.plasma(np.linspace(0, 1, len(x_points)))
    ax.plot(data)

    for j, (x, y) in enumerate(zip(x_points, y_points)):
        ax.scatter(x, y, color=colors[j], s=50, zorder=5)
        y_offset = get_y_offset(j)
        ax.text(x, y + y_offset, f"{x}", fontsize=20, ha="center", va="bottom", color="black")

    ax.set_ylim([0, 0.7])
    ax.set_xlabel("Band Index", fontsize=27)
    ax.set_ylabel("Reflectance", fontsize=27)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)

    ax.text(0.5, -0.3, f"{labels[i]} {titles[i]}", transform=ax.transAxes, fontsize=30, ha="center", va="top")

plt.tight_layout()
plt.savefig("bandc.png", dpi=600)
plt.show()
