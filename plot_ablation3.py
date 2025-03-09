import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

files = ["results/40_bsdr_ad_cnn/summary.csv"]
algorithms = ["bsdrcnn_r","bsdrcnn_r_wo_order"]
modes = ["dyn","dyn"]
algorithm_labels = ["AD-CNN (proposed)","BSDR-CNN"]

df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)

fig, axes = plt.subplots(3, 1, figsize=(len(algorithms)*2+1, 14))
metrics = ["r2", "rmse_o", "rpd_o"]
colors = {"AD-CNN (proposed)": "blue", "AD-FCNN": "cyan", "FD-CNN": "green", "FD-FCNN": "lime", "BSDR-CNN": "red", "BSDR-FCNN": "purple"}
markers = {"AD-CNN (proposed)": ".", "AD-FCNN": "+", "FD-CNN": "o", "FD-FCNN": "*", "BSDR-CNN": "v", "BSDR-FCNN": "x"}
line_styles = {"AD-CNN (proposed)": "-", "AD-FCNN": "--", "FD-CNN": ".", "FD-FCNN": ":", "BSDR-CNN": (0, (3, 1, 1, 1)), "BSDR-FCNN": (0, (4, 2, 2, 1))}
metric_labels = ["$R^2$","RMSE","RPD"]
for i, metric in enumerate(metrics):
    ax = axes[i]

    for index, algorithm in enumerate(algorithms):
        mode = modes[index]
        data = df[(df["algorithm"] == algorithm) & (df["mode"] == mode) & (df["train_size"] == 0.65)]
        algorithm_label = algorithm_labels[index]
        color = colors[algorithm_label]
        marker = markers[algorithm_label]
        line_style = line_styles[algorithm_label]

        ax.plot(data["target_size"], data[metric], color=color,
                marker=marker, markersize=8, label=algorithm_label, linestyle=line_style)

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

fig.legend(algorithm_labels,loc="upper center", bbox_to_anchor=(0.55, 1), fontsize=19)
plt.tight_layout(rect=[0, 0, 1, 0.90])
plt.savefig("ablation3.png", dpi=600)
plt.show()
