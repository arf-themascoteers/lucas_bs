import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("imp_res/ablation2.csv")

data_bsdrcnn = data[(data["algorithm"] == "bsdrcnn_r") & (data["mode"] == "dyn")]
data_bsdrfc = data[(data["algorithm"] == "bsdrfc_r") & (data["mode"] == "dyn")]

data_bsdrcnn2 = data[(data["algorithm"] == "bsdrcnn_r") & (data["mode"] == "static")]
data_bsdrfc2 = data[(data["algorithm"] == "bsdrfc_r") & (data["mode"] == "static")]

data_bsdr = data[(data["algorithm"] == "bsdrfc") & (data["mode"] == "dyn")]

data_bsdrc = data[(data["algorithm"] == "bsdrcnn_r_wo_order") & (data["mode"] == "dyn")]

fig, axes = plt.subplots(3, 1, figsize=(7, 14))
metrics = ["r2", "rmse_o", "rpd_o"]
colors = {"bsdrcnn": "blue", "bsdrfc": "red", "bsdrcnn2": "green", "bsdrfc2": "orange", "bsdr": "purple", "bsdrc": "cyan"}
markers = {"bsdrcnn": ".", "bsdrfc": "+", "bsdrcnn2": "o", "bsdrfc2": "*", "bsdr": "v", "bsdrc": "o"}
line_Style = {"bsdrcnn": "-", "bsdrfc": "--", "bsdrcnn2": "-.", "bsdrfc2": ":", "bsdr": (0, (3, 1, 1, 1)), "bsdrc": (0, (4, 2, 2, 1))}
labels = {"bsdrcnn": "AD-CNN (proposed)", "bsdrfc": "AD-FCNN", "bsdrcnn2": "FD-CNN", "bsdrfc2": "FD-FCNN", "bsdr": "BSDR-FCNN", "bsdrc": "BSDR-CNN"}
metric_labels = ["$R^2$","RMSE","RPD"]
for i, metric in enumerate(metrics):
    ax = axes[i]

    ax.plot(data_bsdrcnn["target_size"], data_bsdrcnn[metric], color=colors["bsdrcnn"],
            marker=markers["bsdrcnn"], markersize=8, label=labels["bsdrcnn"], linestyle=line_Style["bsdrcnn"])
    # ax.plot(data_bsdrfc["target_size"], data_bsdrfc[metric], color=colors["bsdrfc"],
    #         marker=markers["bsdrfc"], markersize=8, label=labels["bsdrfc"], linestyle=line_Style["bsdrfc"])
    # ax.plot(data_bsdrcnn2["target_size"], data_bsdrcnn2[metric], color=colors["bsdrcnn2"],
    #         marker=markers["bsdrcnn2"], markersize=8, label=labels["bsdrcnn2"], linestyle=line_Style["bsdrcnn2"])
    # ax.plot(data_bsdrfc2["target_size"], data_bsdrfc2[metric], color=colors["bsdrfc2"],
    #         marker=markers["bsdrfc2"], markersize=8, label=labels["bsdrfc2"], linestyle=line_Style["bsdrfc2"])
    # ax.plot(data_bsdr["target_size"], data_bsdr[metric], color=colors["bsdr"],
    #         marker=markers["bsdr"], markersize=8, label=labels["bsdr"], linestyle=line_Style["bsdr"])
    ax.plot(data_bsdr["target_size"], data_bsdrc[metric], color=colors["bsdrc"],
            marker=markers["bsdrc"], markersize=8, label=labels["bsdrc"], linestyle=line_Style["bsdrc"])

    ax.set_xscale("log", base=2)
    ax.set_xticks([8, 16, 32, 64, 128, 256, 512])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    ax.set_ylabel(metric_labels[i], fontsize=22)

    ax.set_xlabel("Target Size (log\u2082 scale)", fontsize=22)
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)
    ax.set_yticks(np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 4))

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2f}'))

fig.subplots_adjust(hspace=0.3)
subplot_labels = ['(a) $R^2$', '(b) RMSE', '(c) RPD', '(d) Training time (seconds)']
for i, ax in enumerate(axes.flatten()):
    ax.text(0.5, -0.3, subplot_labels[i], transform=ax.transAxes, fontsize=25, ha='center', va='top')

#fig.legend(loc="upper center", bbox_to_anchor=(0.55, 1), fontsize=19)
plt.tight_layout(rect=[0, 0, 1, 0.90])
plt.savefig("ablation2.png", dpi=600)
plt.show()
