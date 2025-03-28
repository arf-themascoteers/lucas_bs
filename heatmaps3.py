import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

df = pd.read_csv("imp_res/best.csv")
df["train_size"] = (df["train_size"] * 100).astype(int)

metrics = ["r2", "rmse_o", "rpd_o"]
labels = ["$R^2$", "RMSE", "RPD"]

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 18))

for i, (metric, label) in enumerate(zip(metrics, labels)):
    pivot_table = df.pivot(index="train_size", columns="target_size", values=metric)
    sns.heatmap(pivot_table, annot=pivot_table.map(lambda x: f"{x:.2f}"), fmt="", cmap="viridis", ax=axes[i])
    row = pivot_table.index.get_loc(75)
    col = pivot_table.columns.get_loc(128)
    axes[i].add_patch(plt.Rectangle((col, row), 1, 1, fill=False, edgecolor='blue', lw=3))
    axes[i].set_xlabel("Lower dimensional size", fontfamily="Times New Roman", fontsize=17)
    axes[i].set_ylabel("Training size (%)", fontfamily="Times New Roman", fontsize=17)
    axes[i].set_title(label, fontfamily="Times New Roman", fontsize=17)
    axes[i].text(0.5, -0.2, f'({chr(97+i)})', transform=axes[i].transAxes,
                 ha='center', va='center', fontfamily="Times New Roman", fontsize=17)

plt.tight_layout(pad=2)
plt.savefig("heatmaps.png", pad_inches=0.1, bbox_inches='tight',dpi=600)
plt.show()
