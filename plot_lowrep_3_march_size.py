import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from ds_manager import DSManager
plt.rcParams['font.family'] = 'Times New Roman'
bsdr_indices = []
for size in [
f"backup_results/2_compo/bsdrfc_r_norder_lucas_8_robust_dyn_0.75_0.csv",
f"backup_results/3_compo/bsdrfc_r_norder_lucas_16_robust_dyn_0.75_0.csv",
f"backup_results/2_compo/bsdrfc_r_norder_lucas_32_robust_dyn_0.75_0.csv"
]:
    df = pd.read_csv(size)
    last_row = df.iloc[-1]
    values_list = last_row.loc['band_1':].dropna().tolist()
    values_list = [int(i) for i in values_list if i <4200]
    bsdr_indices.append(values_list)

df = pd.read_csv("for_plots/sc/r_wo_order.csv")
last_row = df.iloc[-1]
values_list = last_row.loc['band_1':].dropna().tolist()
values_list = [int(i) for i in values_list]
bsdr_indices.append(values_list)

ad_indices = []
for size in [
f"backup_results/2_compo/bsdrcnn_r_lucas_8_robust_dyn_0.75_0.csv",
f"backup_results/3_compo/bsdrcnn_r_lucas_16_robust_dyn_0.75_0.csv",
f"backup_results/2_compo/bsdrcnn_r_lucas_32_robust_dyn_0.75_0.csv"
]:
    df = pd.read_csv(size)
    last_row = df.iloc[-1]
    values_list = last_row.loc['band_1':].dropna().tolist()
    values_list = [int(i) for i in values_list if i <4200]
    ad_indices.append(values_list)

fig, axes = plt.subplots(3, 2, figsize=(5, 6))

ds = DSManager(name="min_lucas", shuffle=False)
original_data = ds.data[:, 0:-1]

row = original_data[1]
X = row

min_val = np.min(X)
max_val = np.max(X)

row_labels = ["8 bands", "16 bands", "32 bands"]
col_labels = ["BSDR", "Adaptive-downsampling"]

lims = [8,32,128]

for epoch in range(3):
    axes[epoch, 0].set_ylim([min_val, max_val])
    axes[epoch, 1].set_ylim([min_val, max_val])

    # axes[epoch, 0].set_xlim([0, lims[epoch]])
    # axes[epoch, 1].set_xlim([0, lims[epoch]])

    bsdr_i = bsdr_indices[epoch]
    ad_i = ad_indices[epoch]
    h_bsdr = X[bsdr_i]
    h_ad = X[ad_i]

    axes[epoch, 0].plot(h_bsdr, color="orange")
    axes[epoch, 0].set_xlabel("Index", fontsize=10)
    axes[epoch, 0].set_ylabel("Value", fontsize=10)
    axes[epoch, 0].set_title(f"Selected bands: {len(bsdr_i)}", fontsize=11)

    print(len(ad_i))
    axes[epoch, 1].plot(h_ad, color="blue")
    axes[epoch, 1].set_xlabel("Index", fontsize=10)
    axes[epoch, 1].set_ylabel("Value", fontsize=10)
    axes[epoch, 1].set_title(f"Selected bands: {len(ad_i)}", fontsize=11)


    axes[epoch, 1].annotate(row_labels[epoch], xy=(1.1, 0.5), xycoords="axes fraction",
                            fontsize=10, ha="center", va="center", rotation=90)

# Add column titles
#fig.suptitle("Comparison of BSDR and Adaptive-downsampling at Different Training Stages", fontsize=14)
axes[-1,0].text(0.5, -0.8, "(a)", fontsize=11, ha="center", transform=axes[-1,0].transAxes)
axes[-1,1].text(2, -0.8, "(b)", fontsize=11, ha="center", transform=axes[-1,0].transAxes)

plt.tight_layout(pad=2)
plt.savefig("low_rep_3_march_ba_size.png", dpi=600, pad_inches=0)
plt.show()
