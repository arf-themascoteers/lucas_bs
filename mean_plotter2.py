import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16

start = "466|927|1389|1850|2312|2773|3235|3696"
end = "661|879|1398|2023|2315|2794|3466|3605"

data = pd.read_csv("mean.csv").to_numpy()
data = data.reshape(-1)

res = pd.read_csv("results/18_best_75_8_1/bsdrcnn_r_lucas_8_robust_dyn_0.75_0.csv").to_numpy()
res = res[50, -8:]

start = start.split("|")
start = [int(v) for v in start]

end = end.split("|")
end = [int(v) for v in end]

res = res.astype(int).tolist()

indices = [start, res, end]

#print(indices)

titles = ["Initial state", "Midpoint of training", "End of training"]
labels = ["(a)", "(b)", "(c)"]

fig, axes = plt.subplots(3, 1, figsize=(8, 10))

def get_y_offset(j):
    if j%2 == 0:
        return 0.06
    else:
        return -0.12


for i, ax in enumerate(axes):
    x_points = indices[i]
    #print(x_points)
    y_points = data[x_points]
    colors = plt.cm.Set1(np.linspace(0, 1, len(x_points)))
    ax.plot(data, color="black")

    for j, (x, y) in enumerate(zip(x_points, y_points)):
        ax.scatter(x, y, color=colors[j], s=200, zorder=5)
        y_offset = get_y_offset(j)
        ax.text(x, y + y_offset, f"{x}", fontsize=14, ha="center", va="bottom", color="black")

    ax.set_ylim([0.1, 0.55])
    ax.set_xlabel("Band Index", fontsize=14)
    ax.set_ylabel("Reflectance", fontsize=14)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)

    ax.text(0.5, -0.3, f"{labels[i]}", transform=ax.transAxes, fontsize=16, ha="center", va="top",
            fontname="Times New Roman"
            )
plt.subplots_adjust(hspace=0.5, top=0.85)
#plt.tight_layout(pad=0)
plt.savefig("bandc.png", dpi=600, pad_inches=0)
plt.show()
