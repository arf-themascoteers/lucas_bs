import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'Times New Roman'
start = "466|927|1389|1850|2312|2773|3235|3696"
end = "661|879|1398|2023|2315|2794|3466|3605"

data = pd.read_csv("mean.csv").to_numpy()
data = data.reshape(-1)

start = [int(v) for v in start.split("|")]
end = [int(v) for v in end.split("|")]

min_y = np.min(data) - 0.02
max_y = np.max(data) + 0.02

plt.figure(figsize=(6, 2))
plt.ylim([min_y, max_y])
plt.xticks([])
plt.yticks([])
plt.gca().spines[:].set_visible(False)
plt.plot(data, color="black", linewidth=3)
plt.savefig("plot_0.png", bbox_inches=plt.gca().get_window_extent().transformed(plt.gcf().dpi_scale_trans.inverted()), transparent=True)
plt.close()

plt.figure(figsize=(6, 2))
plt.ylim([min_y, max_y])
plt.xticks([])
plt.yticks([])
plt.gca().spines[:].set_visible(False)
x = start
y = data[x]
colors = plt.cm.Set1(np.linspace(0, 1, len(x)))
plt.plot(data, color="black", linewidth=3)
for i in range(len(x)):
    plt.scatter(x[i], y[i], color=colors[i], s=200, zorder=5)
plt.savefig("plot_1.png", bbox_inches=plt.gca().get_window_extent().transformed(plt.gcf().dpi_scale_trans.inverted()), transparent=True)
plt.close()


plt.figure(figsize=(6, 1))
plt.ylim([0, 1])
plt.xlim([0, 4200])
plt.xticks([])
plt.yticks([])
plt.gca().spines[:].set_visible(False)
x = start
y = [0.5]*8
colors = plt.cm.Set1(np.linspace(0, 1, len(x)))
#plt.plot(data, color="black", linewidth=3)
for i in range(len(x)):
    plt.scatter(x[i], y[i], color=colors[i], s=200, zorder=5)
plt.savefig("plot_1_2.png", bbox_inches=plt.gca().get_window_extent().transformed(plt.gcf().dpi_scale_trans.inverted()), transparent=True)
plt.close()

plt.figure(figsize=(6, 2))
plt.ylim([min_y, max_y])
plt.xticks([])
plt.yticks([])
plt.gca().spines[:].set_visible(False)
x = end
y = data[x]
colors = plt.cm.Set1(np.linspace(0, 1, len(x)))
plt.plot(data, color="black", linewidth=3)
for i in range(len(x)):
    plt.scatter(x[i], y[i], color=colors[i], s=200, zorder=5)
plt.savefig("plot_2.png", bbox_inches=plt.gca().get_window_extent().transformed(plt.gcf().dpi_scale_trans.inverted()), transparent=True)
plt.close()

plt.figure(figsize=(6, 1))
plt.ylim([0, 1])
plt.xlim([0, 4200])
plt.xticks([])
plt.yticks([])
plt.gca().spines[:].set_visible(False)
x = end
y = [0.5]*8
colors = plt.cm.Set1(np.linspace(0, 1, len(x)))
#plt.plot(data, color="black", linewidth=3)
for i in range(len(x)):
    plt.scatter(x[i], y[i], color=colors[i], s=200, zorder=5)
plt.savefig("plot_2_2.png", bbox_inches=plt.gca().get_window_extent().transformed(plt.gcf().dpi_scale_trans.inverted()), transparent=True)
plt.close()


plt.figure(figsize=(6, 2))
plt.ylim([min_y, max_y])
plt.xticks([])
plt.yticks([])
plt.gca().spines[:].set_visible(False)
x = end
y = data[x]
colors = plt.cm.Set1(np.linspace(0, 1, len(x)))
plt.plot(x, y, color="black", linewidth=3)
for i in range(len(x)):
    plt.scatter(x[i], y[i], color=colors[i], s=200, zorder=5)
plt.savefig("plot_3.png", bbox_inches=plt.gca().get_window_extent().transformed(plt.gcf().dpi_scale_trans.inverted()), transparent=True)
plt.close()

plt.figure(figsize=(6, 2))
plt.ylim([min_y, max_y])
plt.xticks([])
plt.yticks([])
plt.gca().spines[:].set_visible(False)
x = start
y = data[x]
colors = plt.cm.Set1(np.linspace(0, 1, len(x)))
plt.plot(x, y, color="black", linewidth=3)
for i in range(len(x)):
    plt.scatter(x[i], y[i], color=colors[i], s=200, zorder=5)
plt.savefig("plot_4.png", bbox_inches=plt.gca().get_window_extent().transformed(plt.gcf().dpi_scale_trans.inverted()), transparent=True)
plt.close()
