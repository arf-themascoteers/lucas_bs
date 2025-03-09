import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("back/check1/summary.csv")
data_bsdrcnn = data[(data["algorithm"] == "bsdrcnn") & (data["mode"] == "dyn")]
data_bsdrfc = data[(data["algorithm"] == "bsdrfc") & (data["mode"] == "dyn")]

data_bsdrcnn2 = data[(data["algorithm"] == "bsdrcnn") & (data["mode"] == "static")]
data_bsdrfc2 = data[(data["algorithm"] == "bsdrfc") & (data["mode"] == "static")]

tdf1 = pd.read_csv("backup_results/model_time/summary.csv")
tdf2 = pd.read_csv("backup_results/model_time_static/summary.csv")


data_bsdrcnn = data_bsdrcnn.merge(tdf1[['algorithm', 'mode', 'target_size', 'execution_time']], on=['algorithm', 'mode', 'target_size'], how='left')
data_bsdrfc = data_bsdrfc.merge(tdf1[['algorithm', 'mode', 'target_size', 'execution_time']], on=['algorithm', 'mode', 'target_size'], how='left')
data_bsdrcnn2 = data_bsdrcnn2.merge(tdf2[['algorithm', 'mode', 'target_size', 'execution_time']], on=['algorithm', 'mode', 'target_size'], how='left')
data_bsdrfc2 = data_bsdrfc2.merge(tdf2[['algorithm', 'mode', 'target_size', 'execution_time']], on=['algorithm', 'mode', 'target_size'], how='left')


fig, axes = plt.subplots(2, 2, figsize=(12, 11))
metrics = ["r2", "rmse", "rpd", "execution_time"]
colors = {"bsdrcnn": "blue", "bsdrfc": "red", "bsdrcnn2": "green", "bsdrfc2": "orange"}
markers = {"bsdrcnn": ".", "bsdrfc": "+", "bsdrcnn2": "o", "bsdrfc2": "*"}
line_Style = {"bsdrcnn": "-", "bsdrfc": "--", "bsdrcnn2": "-.", "bsdrfc2": ":"}
labels = {"bsdrcnn": "AD-CNN (proposed)", "bsdrfc": "AD-FCNN", "bsdrcnn2": "FD-CNN", "bsdrfc2": "FD-FCNN"}

for i, metric in enumerate(metrics):
    ax = axes[i // 2, i % 2]
    if i == 3:
        ax.set_yscale('log')
    ax.plot(data_bsdrcnn["target_size"], data_bsdrcnn[metric], color=colors["bsdrcnn"],
            marker=markers["bsdrcnn"], markersize=8, label=labels["bsdrcnn"], linestyle=line_Style["bsdrcnn"])
    ax.plot(data_bsdrfc["target_size"], data_bsdrfc[metric], color=colors["bsdrfc"],
            marker=markers["bsdrfc"], markersize=8, label=labels["bsdrfc"], linestyle=line_Style["bsdrfc"])
    ax.plot(data_bsdrcnn2["target_size"], data_bsdrcnn2[metric], color=colors["bsdrcnn2"],
            marker=markers["bsdrcnn2"], markersize=8, label=labels["bsdrcnn2"], linestyle=line_Style["bsdrcnn2"])
    ax.plot(data_bsdrfc2["target_size"], data_bsdrfc2[metric], color=colors["bsdrfc2"],
            marker=markers["bsdrfc2"], markersize=8, label=labels["bsdrfc2"], linestyle=line_Style["bsdrfc2"])

    ax.set_xscale("log", base=2)
    ax.set_xticks([8, 16, 32, 64, 128, 256, 512])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    if metric == "r2":
        ax.set_ylabel(r"$R^2$", fontsize=22)
    if metric == "execution_time":
        ax.set_ylabel("Time (log\u2081\u2080 scale)", fontsize=22)
    else:
        ax.set_ylabel(metric.upper(), fontsize=22)

    ax.set_xlabel("Target Size (log\u2082 scale)", fontsize=22)
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)
    ax.set_yticks(np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 4))
    if i == 3:
        ymin = 4
        ymax = 10001
        ax.set_yticks([10 ** i for i in range(int(np.log10(4)), int(np.log10(ymax)) + 1)])
        ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda y, _: f'$10^{{{int(np.log10(y))}}}$'))
    else:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2f}'))

fig.subplots_adjust(hspace=0.3)
subplot_labels = ['(a) $R^2$', '(b) RMSE', '(c) RPD', '(d) Training time (seconds)']
for i, ax in enumerate(axes.flatten()):
    ax.text(0.5, -0.3, subplot_labels[i], transform=ax.transAxes, fontsize=25, ha='center', va='top')

fig.legend(labels.values(), loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1), fontsize=19)
plt.tight_layout(rect=[0, 0, 1, 0.90])
plt.savefig("ablation1.png", dpi=600)
plt.show()
