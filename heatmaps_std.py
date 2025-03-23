import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm
import os

# df = pd.read_csv("results/13_best/details.csv")
# df["train_size"] = (df["train_size"] * 100).astype(int)
#
# cols = ["r2", "rmse", "rpd", "rpiq", "r2_o", "rmse_o", "rpd_o", "rpiq_o","execution_time"]
# df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
# grouped = df.groupby(["target_size", "train_size"])[cols].agg(["mean", "std"]).reset_index()
# grouped.columns = ['_'.join(col).strip('_') for col in grouped.columns.values]
# grouped.to_csv("output.csv", index=False)


df = pd.read_csv("output.csv")

metrics = ["r2_std", "rmse_o_std", "rpd_o_std", "execution_time_std"]
labels = ["$R^2$", "RMSE", "RPD", "Execution time"]

for i,metric in enumerate(metrics):
    pivot_table = df.pivot(index="train_size", columns="target_size", values=metric)
    plt.figure(figsize=(8, 6))
    if metric == "execution_time":
        sns.heatmap(pivot_table, annot=True, cmap="viridis",
                    norm=LogNorm(vmin=pivot_table.min().min(), vmax=pivot_table.max().max()),
                    fmt=",.0f", annot_kws={"size": 10})



    else:
        sns.heatmap(pivot_table, annot=pivot_table.map(lambda x: f"{x:.2f}"), fmt="", cmap="viridis")

    row = pivot_table.index.get_loc(75)
    col = pivot_table.columns.get_loc(128)
    plt.gca().add_patch(plt.Rectangle((col, row), 1, 1, fill=False, edgecolor='blue', lw=3))

    plt.xlabel("Lower-dimensional size", fontsize=14)
    plt.ylabel("Training size (%)", fontsize=14)
    #plt.title(metric)
    plt.tight_layout(pad=0)
    os.makedirs("heatmaps3", exist_ok=True)
    plt.savefig("heatmaps3/" + metric + ".png", pad_inches=0, bbox_inches='tight')
    plt.show()
