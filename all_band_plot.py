import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('imp_res/all_bands_m2.csv')

fig, axes = plt.subplots(2, 2, figsize=(6,6))

metrics = ['r2', 'rmse_o', 'rpd_o', 'execution_time']
titles = ['(a) $R^2$', ' (b) RMSE', '(c) RPD', '(d) Execution time (seconds)']

axes[1, 1].set_yscale('log')

names = {
    "bsdrcnn_r_4200_2" : "All 4,200 bands",
    "bsdrcnn_r" : "128 bands (AD-CNN)",
}

colors = {
    "bsdrcnn_r_4200_2" : "red",
    "bsdrcnn_r" : "blue",
}


for i, ax in enumerate(axes.flat):
    for algo in df['algorithm'].unique():
        subset = df[df['algorithm'] == algo]
        ax.plot(
            subset['train_size'],
            subset[metrics[i]],
            marker='o',
            label=names[algo],
            color=colors[algo]
        )
    ax.set_xlabel('Train Size')

axes[0, 0].set_ylabel('$R^2$')
axes[0, 1].set_ylabel('RMSE')
axes[1, 0].set_ylabel('RPD')
axes[1, 1].set_ylabel(r"Seconds ($\log_{10}$ scale)")


for ax, title in zip(axes.flat, titles):
    ax.set_xticks([0.05, 0.25, 0.45, 0.65, 0.85])
    ax.set_xlabel("Training size (%)", fontsize=10)
    ax.text(0.5, -0.35, title, transform=ax.transAxes, fontsize=12, ha='center')


plt.draw()

#fig.subplots_adjust(hspace=0.6)
fig.legend(names.values(), loc='upper center', ncol=2, fontsize=10, frameon=True)


plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig("all_bands.png", pad_inches=0)
plt.show()