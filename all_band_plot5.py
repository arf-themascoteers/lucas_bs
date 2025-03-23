import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

df = pd.read_csv('imp_res/all_bands_m4.csv')

adcnn_data = pd.read_csv("imp_res/best.csv")
adcnn_data = adcnn_data[
    (adcnn_data["algorithm"] == "bsdrcnn_r") &
    (adcnn_data["target_size"].isin([128, 256, 512])) &
    (adcnn_data["mode"] == "dyn")
]


df = pd.concat((df, adcnn_data), ignore_index=True)


fig, axes = plt.subplots(2, 2, figsize=(6,6))

metrics = ['r2', 'rmse_o', 'rpd_o', 'execution_time']
titles = ['(a)', '(b)', '(c)', '(d)']

axes[1, 1].set_yscale('log')

algorithms = [
    {
        "name":"bsdrcnn_r_4200_3",
        "label":"1D-CNN with all 4,200 bands",
        "target_size":4200,
        "color":"red",
        "marker":"+",
        "linestyle":"--",
    },
    {
        "name": "bsdrcnn_r",
        "label": "AD-CNN with 128 bands (proposed)",
        "target_size": 128,
        "color": "blue",
        "marker": ".",
        "linestyle": "-",
    },
    {
        "name": "bsdrcnn_r",
        "label": "AD-CNN with 256 bands",
        "target_size": 256,
        "color": "green",
        "marker": "o",
        "linestyle": "-.",
    },
    {
        "name": "bsdrcnn_r",
        "label": "AD-CNN with 512 bands",
        "target_size": 512,
        "color": "purple",
        "marker": "*",
        "linestyle": ":",
    }
]

for i, ax in enumerate(axes.flat):
    for algo_index,algo in enumerate(algorithms):
        name = algo["name"]
        label = algo["label"]
        target_size = algo["target_size"]
        color = algo["color"]
        linestyle = algo["linestyle"]
        marker = algo["marker"]
        subset = df[(df['algorithm'] == name)&(df['target_size'] == target_size)]
        ax.plot(
            subset['train_size'],
            subset[metrics[i]],
            marker=marker,
            linestyle=linestyle,
            label=label,
            color=color
        )
    ax.set_xlabel('Train Size')

axes[0, 0].set_ylabel('$R^2$')
axes[0, 1].set_ylabel('RMSE')
axes[1, 0].set_ylabel('RPD')
axes[1, 1].set_ylabel("Seconds (log$_{10}$ scale)", fontname="Times New Roman")

xticks = df['train_size'].unique()
xtick_labels = [int(x * 100) for x in xticks]

for ax, title in zip(axes.flat, titles):
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)
    ax.set_xlabel("Training size (%)")
    ax.text(0.5, -0.42, title, transform=ax.transAxes, ha='center', fontname="Times New Roman")

plt.draw()

names = [a["label"] for a in algorithms]
fig.legend(names, loc='upper center', ncol=2, frameon=True)


plt.tight_layout(rect=[0, 0, 1, 0.88], pad=0.2)
plt.savefig("all_bands5.png", pad_inches=0, dpi=600)
plt.show()