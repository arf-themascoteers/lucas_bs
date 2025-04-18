import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

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
plt.xlim([0, 4200])
plt.xticks([])
plt.yticks([])
plt.gca().spines[:].set_visible(False)
plt.plot(data, color="black", linewidth=3)
plt.savefig("plot_0.png", bbox_inches=plt.gca().get_window_extent().transformed(plt.gcf().dpi_scale_trans.inverted()), transparent=True)
plt.close()

plt.figure(figsize=(6, 2))
plt.ylim([min_y, max_y])
plt.xlim([0, 4200])
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
plt.xlim([0, 4200])
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
plt.xlim([0, 4200])
plt.xticks([])
plt.yticks([])
plt.gca().spines[:].set_visible(False)
prev_end = end
end = [900,1200,500,2300,1800,3500,3100,2600]
x = end
y = data[x]
data_new = data.copy()
data_new[prev_end] = data[prev_end]
colors = plt.cm.Set1(np.linspace(0, 1, len(x)))
plt.plot(data, color="black", linewidth=3)
for i in range(len(x)):
    plt.scatter(x[i], y[i], color=colors[i], s=200, zorder=5)
plt.savefig("plot_2_3.png", bbox_inches=plt.gca().get_window_extent().transformed(plt.gcf().dpi_scale_trans.inverted()), transparent=True)
plt.close()

plt.figure(figsize=(6, 1))
plt.ylim([0, 1])
plt.xlim([0, 4200])
plt.xticks([])
plt.yticks([])
plt.gca().spines[:].set_visible(False)
end = [900,1200,500,2300,1800,3500,3100,2600]
x = end
y = [0.5]*8
colors = plt.cm.Set1(np.linspace(0, 1, len(x)))
#plt.plot(data, color="black", linewidth=3)
#random.shuffle(end)
for i in range(len(x)):
    plt.scatter(x[i], y[i], color=colors[i], s=200, zorder=5)
plt.savefig("plot_2_4.png", bbox_inches=plt.gca().get_window_extent().transformed(plt.gcf().dpi_scale_trans.inverted()), transparent=True)
plt.close()

plt.figure(figsize=(6, 1))
plt.ylim([-0.1, 0.9])
plt.xlim([0, 4200])
plt.xticks([])
plt.yticks([])
plt.gca().spines[:].set_visible(False)
end = [900,1200,500,2300,1800,3500,3100,2600]
indices = np.argsort(end)
x = end
y = [0.1,0.3,0.5,0.2,0.7,0.3,0.6,0.1]
x.sort()
colors = plt.cm.Set1(np.linspace(0, 1, len(x)))

plt.plot(x,y, color="black", linewidth=3)
for i in range(len(x)):
    plt.scatter(x[i], y[i], color=colors[indices[i]], s=200, zorder=5)
plt.savefig("plot_2_5.png", bbox_inches=plt.gca().get_window_extent().transformed(plt.gcf().dpi_scale_trans.inverted()), transparent=True)
plt.close()

plt.figure(figsize=(6, 2))
plt.ylim([min_y, max_y])
plt.xlim([0, 4200])
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
plt.xlim([0, 4200])
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


high = np.random.uniform(0.5, 1.0, size=10)
low = np.random.uniform(0.0, 0.1, size=90)
values = np.concatenate([high, low])
np.random.shuffle(values)
plt.figure(figsize=(6, 2))
plt.ylim([-0.1, 1.1])
plt.xlim([-1, 101])
plt.xticks([])
plt.yticks([])
plt.gca().spines[:].set_visible(False)
colors = plt.cm.Set1(np.linspace(0, 1, 100))
for i in range(100):
    plt.bar(i, values[i], color=colors[i])
plt.savefig("plot_7.png", bbox_inches=plt.gca().get_window_extent().transformed(plt.gcf().dpi_scale_trans.inverted()), transparent=True)
plt.close()


plt.figure(figsize=(6, 2))
plt.ylim([-0.1, 1.1])
plt.xlim([-1, 101])
plt.xticks([])
plt.yticks([])
plt.gca().spines[:].set_visible(False)
colors = plt.cm.Set1(np.linspace(0, 1, 100))
downscaled = data.reshape(100, 42).mean(axis=1)
recal = downscaled * values
for i in range(100):
    plt.bar(i, recal[i], color=colors[i])
plt.savefig("plot_8.png", bbox_inches=plt.gca().get_window_extent().transformed(plt.gcf().dpi_scale_trans.inverted()), transparent=True)
plt.close()

