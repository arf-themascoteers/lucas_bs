import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import get_cmap
from scipy.interpolate import interp1d

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 17

df = pd.read_csv("results/6_ad_8/bsdrcnn_r_lucas_8_robust_dyn_0.75_0.csv")

r2 = df["r2"].tolist()
rmse = df["rmse"].tolist()
rpd = df["rpd"].tolist()

x_old = np.linspace(0, 1, len(r2))
x_new = np.linspace(0, 1, 1000)




band_cols = [col for col in df.columns if col.startswith("band_")]
band_values = df[band_cols].values.tolist()
epochs = np.arange(1, len(r2) + 1)
fig, (ax4, ax1) = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'width_ratios': [1, 1]})

# r2 = interp1d(x_old, r2, kind='cubic')(x_new)
# rmse = interp1d(x_old, rmse, kind='cubic')(x_new)
# rpd = interp1d(x_old, rpd, kind='cubic')(x_new)
#
#
# band_array = np.array(band_values)
# x_old = np.linspace(0, 1, band_array.shape[0])
# x_new = np.linspace(0, 1, 1000)
#
# band_values = interp1d(x_old, band_array, axis=0, kind='linear')(x_new)
#


# Plot metrics
ax2 = ax1.twinx()
ax3 = ax1.twinx()
ax3.spines["right"].set_position(("axes", 1.15))

ax1.plot(epochs, rmse, 'b-', label='RMSE')
ax2.plot(epochs, r2, 'g--', label='$R^2$')
ax3.plot(epochs, rpd, 'r-.', label='RPD')

ax1.set_xlabel('Epoch')
ax1.set_ylabel('RMSE', color='b')
ax2.set_ylabel('$R^2$', color='g')
ax3.set_ylabel('RPD', color='r')

ax1.tick_params(axis='y', colors='b')
ax2.tick_params(axis='y', colors='g')
ax3.tick_params(axis='y', colors='r')

lines = ax1.get_lines() + ax2.get_lines() + ax3.get_lines()
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.65, 0.7))

cmap = plt.colormaps['viridis']
colors = cmap(np.linspace(0, 1, len(band_values[0])))


for i, bands in enumerate(band_values):
    for j, band in enumerate(bands):
        ax4.scatter(epochs[i], band, s=6, color=colors[j])

ax4.set_xlabel('Epoch')
ax4.set_ylabel('Selected bands')
#ax4.set_title('Band Selection Over Epochs')

ax1.set_xticklabels((ax1.get_xticks() * 10).astype(int))
ax4.set_xticklabels((ax4.get_xticks() * 10).astype(int))

ax3.spines["right"].set_position(("axes", 1.25))

ax4.text(0.5, -0.2, '(a)', transform=ax4.transAxes, ha='center', va='top')
ax1.text(0.5, -0.2, '(b)', transform=ax1.transAxes, ha='center', va='top')

plt.tight_layout()
plt.savefig("epoch.png",dpi=600)
plt.show()
