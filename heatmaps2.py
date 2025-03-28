import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm
import os

df = pd.read_csv("imp_res/best.csv")
df["train_size"] = (df["train_size"] * 100).astype(int)

metrics = ["r2", "rmse_o", "rpd_o", "execution_time"]
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

    plt.xlabel("Lower dimensional size", fontsize=14)
    plt.ylabel("Training size (%)", fontsize=14)
    #plt.title(metric)
    plt.tight_layout(pad=0)
    os.makedirs("heatmaps2", exist_ok=True)
    plt.savefig("heatmaps2/" + metric + ".png", pad_inches=0, bbox_inches='tight')
    plt.show()
