import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'Times New Roman'
start = "466|927|1389|1850|2312|2773|3235|3696"
end = "661|879|1398|2023|2315|2794|3466|3605"


start = start.split("|")
start = [int(v) for v in start]

end = end.split("|")
end = [int(v) for v in end]

fig, axes = plt.subplots(1, 2, figsize=(12, 1))

y_points = [0,0,0,0,0,0,0,0]

for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim([0, 4200])
    for spine in ax.spines.values():
        spine.set_visible(False)


ax = axes[0]
x_points = start
colors = plt.cm.Set1(np.linspace(0, 1, len(x_points)))
for j, (x, y) in enumerate(zip(x_points, y_points)):
    ax.scatter(x, y, color=colors[j], s=200, zorder=5)
    print(x,y)

ax = axes[1]
x_points = end
colors = plt.cm.Set1(np.linspace(0, 1, len(x_points)))
for j, (x, y) in enumerate(zip(x_points, y_points)):
    ax.scatter(x, y, color=colors[j], s=200, zorder=5)

for i, ax in enumerate(axes):
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(f"numberline_{i}.png", bbox_inches=extent, transparent=True)

plt.show()


