import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
data = pd.read_csv("imp_res/ablation_final.csv")

adcnn_data = pd.read_csv("imp_res/best.csv")
adcnn_data = adcnn_data[
    (adcnn_data["algorithm"] == "bsdrcnn_r") & (adcnn_data["train_size"] == 0.75) & (adcnn_data["mode"] == "dyn")]

data = pd.concat((data, adcnn_data), ignore_index=True)

# Filter data
data_bsdrcnn = data[(data["algorithm"] == "bsdrcnn_r") & (data["mode"] == "dyn")]
data_bsdrfc = data[(data["algorithm"] == "bsdrfc_r") & (data["mode"] == "dyn")]

data_bsdrcnn2 = data[(data["algorithm"] == "bsdrcnn_r") & (data["mode"] == "static")]
data_bsdrfc2 = data[(data["algorithm"] == "bsdrfc_r") & (data["mode"] == "static")]

data_bsdr = data[(data["algorithm"] == "bsdrfc") & (data["mode"] == "dyn")]

# Plot settings
metrics = ["r2", "rmse_o", "rpd_o"]
metric_labels = ["$R^2$", "RMSE", "RPD"]
file_names = ["plot_r2.png", "plot_rmse.png", "plot_rpd.png"]
subplot_labels = ["(a) $R^2$", "(b) RMSE", "(c) RPD"]

colors = {"bsdrcnn": "blue", "bsdrfc": "red", "bsdrcnn2": "green", "bsdrfc2": "orange", "bsdr": "purple"}
markers = {"bsdrcnn": ".", "bsdrfc": "+", "bsdrcnn2": "o", "bsdrfc2": "*", "bsdr": "v"}
line_Style = {"bsdrcnn": "-", "bsdrfc": "--", "bsdrcnn2": "-.", "bsdrfc2": ":", "bsdr": (0, (3, 1, 1, 1))}
labels = {
    "bsdr": "BSDR",
    "bsdrfc2": "FD-FCNN",
    "bsdrcnn2": "FD-CNN",
    "bsdrfc": "AD-FCNN",
    "bsdrcnn": "AD-CNN (proposed)"
}

# Generate separate plots
for i, metric in enumerate(metrics):
    if i ==0:
        plt.figure(figsize=(7, 5))
    else:
        plt.figure(figsize=(7, 5))
    plt.plot(data_bsdr["target_size"], data_bsdr[metric], color=colors["bsdr"],
             marker=markers["bsdr"], markersize=8, label=labels["bsdr"], linestyle=line_Style["bsdr"])
    plt.plot(data_bsdrfc2["target_size"], data_bsdrfc2[metric], color=colors["bsdrfc2"],
             marker=markers["bsdrfc2"], markersize=8, label=labels["bsdrfc2"], linestyle=line_Style["bsdrfc2"])
    plt.plot(data_bsdrcnn2["target_size"], data_bsdrcnn2[metric], color=colors["bsdrcnn2"],
             marker=markers["bsdrcnn2"], markersize=8, label=labels["bsdrcnn2"], linestyle=line_Style["bsdrcnn2"])
    plt.plot(data_bsdrfc["target_size"], data_bsdrfc[metric], color=colors["bsdrfc"],
             marker=markers["bsdrfc"], markersize=8, label=labels["bsdrfc"], linestyle=line_Style["bsdrfc"])
    plt.plot(data_bsdrcnn["target_size"], data_bsdrcnn[metric], color=colors["bsdrcnn"],
             marker=markers["bsdrcnn"], markersize=8, label=labels["bsdrcnn"], linestyle=line_Style["bsdrcnn"])

    plt.xscale("log", base=2)
    plt.xticks([8, 16, 32, 64, 128, 256, 512])
    plt.gca().get_xaxis().set_major_formatter(plt.ScalarFormatter())

    plt.ylabel(metric_labels[i], fontsize=22)
    plt.xlabel("Target Size (log₂ scale)", fontsize=22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.yticks(np.linspace(plt.ylim()[0], plt.ylim()[1], 4))
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2f}'))

    #plt.text(0.5, -0.3, subplot_labels[i], transform=plt.gca().transAxes, fontsize=25, ha='center', va='top')

    #if i == 0:
    #    plt.legend(labels.values(), loc="upper left", bbox_to_anchor=(-0.05, 1.4), ncol=2, fontsize=19, borderaxespad=0)


    plt.tight_layout(pad=0.2)
    plt.savefig(file_names[i], dpi=600)
    plt.close()

