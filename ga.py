import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'Times New Roman'
start = "466|927|1389|1850|2312|2773|3235|3696"
end = "661|879|1398|2023|2315|2794|3466|3605"

data = pd.read_csv("mean.csv").to_numpy()
data = data.reshape(-1)

start = start.split("|")
start = [int(v) for v in start]

end = end.split("|")
end = [int(v) for v in end]

titles = ["Original data", "After adaptive-downsampling"]

fig, axes = plt.subplots(1, 4, figsize=(24, 6))

min_y = np.min(data) - 0.02
max_y = np.max(data) + 0.02

for ax in axes:
    ax.set_ylim([min_y, max_y])
    # ax.set_xlabel("Band Index", fontsize=14)
    # ax.set_ylabel("Reflectance", fontsize=14)
    # ax.tick_params(axis='x', labelsize=14)
    # ax.tick_params(axis='y', labelsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

axes[0].plot(data, color="black",linewidth=3)

# ax = axes[1]
# x_points = start
# y_points = data[x_points]
# colors = plt.cm.Set1(np.linspace(0, 1, len(x_points)))
# ax.plot(data, color="black")
# for j, (x, y) in enumerate(zip(x_points, y_points)):
#     ax.scatter(x, y, color=colors[j], s=200, zorder=5)

ax = axes[1]
x_points = start
y_points = data[x_points]
colors = plt.cm.Set1(np.linspace(0, 1, len(x_points)))
ax.plot(data, color="black",linewidth=3)

for j, (x, y) in enumerate(zip(x_points, y_points)):
    ax.scatter(x, y, color=colors[j], s=200, zorder=5)

ax = axes[2]
x_points = end
y_points = data[x_points]
colors = plt.cm.Set1(np.linspace(0, 1, len(x_points)))
ax.plot(data, color="black",linewidth=3)

for j, (x, y) in enumerate(zip(x_points, y_points)):
    ax.scatter(x, y, color=colors[j], s=200, zorder=5)


ax = axes[3]
x_points = end
y_points = data[x_points]
colors = plt.cm.Set1(np.linspace(0, 1, len(x_points)))
ax.plot(x_points, y_points, color="black",linewidth=3)
for j, (x, y) in enumerate(zip(x_points, y_points)):
    ax.scatter(x, y, color=colors[j], s=200, zorder=5)

for i, ax in enumerate(axes):
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(f"subplot_{i}.png", bbox_inches=extent, transparent=True)

plt.show()


