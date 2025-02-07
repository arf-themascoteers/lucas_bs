import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("ad.csv")
data_bsdrcnn = data[data["algorithm"] == "bsdrcnn"]
data_bsdrfc = data[data["algorithm"] == "bsdrfc"]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
metrics = ["r2", "rmse", "rpd", "rpiq"]
colors = {"bsdrcnn": "blue", "bsdrfc": "red"}
markers = {"bsdrcnn": ".", "bsdrfc": "+"}
labels = {"bsdrcnn": "AD-CNN", "bsdrfc": "AD-FCNN"}

for i, metric in enumerate(metrics):
    ax = axes[i // 2, i % 2]
    ax.plot(data_bsdrcnn["target_size"], data_bsdrcnn[metric], color=colors["bsdrcnn"],
            marker=markers["bsdrcnn"], markersize=8, label=labels["bsdrcnn"])
    ax.plot(data_bsdrfc["target_size"], data_bsdrfc[metric], color=colors["bsdrfc"],
            marker=markers["bsdrfc"], markersize=8, label=labels["bsdrfc"])

    best_bsdrcnn = data_bsdrcnn.loc[data_bsdrcnn[metric].idxmax()] if metric != "rmse" else data_bsdrcnn.loc[
        data_bsdrcnn[metric].idxmin()]
    best_bsdrfc = data_bsdrfc.loc[data_bsdrfc[metric].idxmax()] if metric != "rmse" else data_bsdrfc.loc[
        data_bsdrfc[metric].idxmin()]

    ax.scatter(best_bsdrcnn["target_size"], best_bsdrcnn[metric], color=colors["bsdrcnn"], s=100, edgecolor="black",
               zorder=5)
    ax.scatter(best_bsdrfc["target_size"], best_bsdrfc[metric], color=colors["bsdrfc"], s=100, edgecolor="black",
               zorder=5)

    ax.set_xscale("log", base=2)
    ax.set_xticks([8, 16, 32, 64, 128, 256, 512])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    if metric == "r2":
        ax.set_title(r"$R^2$", fontsize=17)
        ax.set_ylabel(r"$R^2$", fontsize=15)
    else:
        ax.set_title(metric.upper(), fontsize=17)
        ax.set_ylabel(metric.upper(), fontsize=15)
    ax.set_xlabel("Target Size (log\u2082 scale)",fontsize=15)

    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)


fig.legend(labels.values(), loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1), fontsize=20)
plt.tight_layout(rect=[0, 0, 1, 0.90])
plt.savefig("ablation1.png", dpi=300)
plt.show()
