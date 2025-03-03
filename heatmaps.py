import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("results/de6/summary3.csv")

metrics = ["r2", "rmse", "rpiq", "rpd", "execution_time"]

for metric in metrics:
    pivot_table = df.pivot(index="train_size", columns="target_size", values=metric)
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot_table, annot=pivot_table.map(lambda x: f"{x:.3f}"), fmt="", cmap="viridis")
    plt.xlabel("target_size")
    plt.ylabel("train_size")
    plt.title(metric)
    plt.savefig("heatmaps/" + metric + ".png")
    plt.show()
