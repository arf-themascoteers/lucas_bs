import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

df = pd.read_csv("imp_res/best.csv")
df["train_size"] = (df["train_size"] * 100).astype(int)

metric = "execution_time"
label = "Execution time"

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))

pivot_table = df.pivot(index="train_size", columns="target_size", values=metric)
sns.heatmap(pivot_table, annot=True, cmap="viridis",
            norm=LogNorm(vmin=pivot_table.min().min(), vmax=pivot_table.max().max()),
            fmt=",.0f")  # Adds commas every 3 digits


row = pivot_table.index.get_loc(75)
col = pivot_table.columns.get_loc(128)
axes.add_patch(plt.Rectangle((col, row), 1, 1, fill=False, edgecolor='blue', lw=3))
axes.set_xlabel("Lower dimensional size", fontfamily="Times New Roman", fontsize=17)
axes.set_ylabel("Training size (%)", fontfamily="Times New Roman", fontsize=17)
axes.set_title(label, fontfamily="Times New Roman", fontsize=17)

plt.tight_layout(pad=0)
plt.savefig("heatmap_time.png", pad_inches=0, bbox_inches='tight',dpi=600)
plt.show()
