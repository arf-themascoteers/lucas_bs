import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from ds_manager import DSManager

plt.rcParams['font.family'] = 'Times New Roman'

bsdr_indices = []
df = pd.read_csv("backup_results/2_compo/bsdrfc_r_norder_lucas_128_robust_dyn_0.75_0.csv")
for i in range(3):
    last_row = df.iloc[i+1]
    values_list = last_row.loc['band_1':].dropna().tolist()
    values_list = [int(i) for i in values_list]
    values_list = np.array(values_list)
    bsdr_indices.append(values_list)


ad_indices = []
df = pd.read_csv("backup_results/1_cnnr/bsdrcnn_r_lucas_128_robust_dyn_0.75_0.csv")
for i in range(4):
    last_row = df.iloc[i+1]
    values_list = last_row.loc['band_1':].dropna().tolist()
    values_list = [int(i) for i in values_list]
    ad_indices.append(values_list)

fd_indices = np.linspace(0, 4199, 128, dtype=int)

bsdr_indices.insert(0, fd_indices)
ad_indices.insert(0, fd_indices)

fig, axes = plt.subplots(3, 2, figsize=(5, 5))

ds = DSManager(name="min_lucas", shuffle=False)
original_data = ds.data[:, 0:-1]

row = original_data[1]
X = row

min_val = np.min(X)-0.02
max_val = np.max(X)+0.02

row_labels = ["Start", "Min-point", "End"]
col_labels = ["BSDR", "Adaptive-downsampling"]

for epoch in range(3):
    axes[epoch, 0].set_ylim([min_val, max_val])
    axes[epoch, 1].set_ylim([min_val, max_val])

    bsdr_i = bsdr_indices[epoch]
    ad_i = ad_indices[epoch]

    h_bsdr = X[bsdr_i]
    h_ad = X[ad_i]

    axes[epoch, 0].plot(h_bsdr, color="orange")
    axes[epoch, 0].set_xlabel("Index", fontsize=11)
    axes[epoch, 0].set_ylabel("Value", fontsize=11)

    axes[epoch, 1].plot(h_ad, color="blue")
    axes[epoch, 1].set_xlabel("Index", fontsize=11)
    axes[epoch, 1].set_ylabel("Value", fontsize=11)

    # Add row labels on the left side
    axes[epoch, 1].annotate(row_labels[epoch], xy=(1.2, 0.5), xycoords="axes fraction",
                            fontsize=11, ha="center", va="center", rotation=90)
    axes[epoch, 0].tick_params(axis='both', labelsize=11)
    axes[epoch, 1].tick_params(axis='both', labelsize=11)

# Add column titles
#fig.suptitle("Comparison of BSDR and Adaptive-downsampling at Different Training Stages", fontsize=14)
axes[-1,0].text(0.5, -0.58, "(a)", fontsize=11, ha="center", transform=axes[-1,0].transAxes)
axes[-1,1].text(2, -0.58, "(b)", fontsize=11, ha="center", transform=axes[-1,0].transAxes)

plt.subplots_adjust(hspace=0.6, wspace=0.5)
#plt.tight_layout(rect=[0, 0, 1, 1], pad=0)
plt.savefig("low_rep_3_march_ba_labeled.png", dpi=600)
plt.show()

