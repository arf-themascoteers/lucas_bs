import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from ds_manager import DSManager

bsdr_indices = []
df = pd.read_csv("for_plots/sc/r_wo_order.csv")
for i in range(4):
    last_row = df.iloc[i+1]
    values_list = last_row.loc['band_1':].dropna().tolist()
    values_list = [int(i) for i in values_list]
    print(len(values_list))
    bsdr_indices.append(values_list)

ad_indices = []
df = pd.read_csv("results/1_cnnr/bsdrcnn_r_lucas_128_robust_dyn_0.75_0.csv")
for i in range(4):
    last_row = df.iloc[i+1]
    values_list = last_row.loc['band_1':].dropna().tolist()
    values_list = [int(i) for i in values_list]
    print(len(values_list))
    ad_indices.append(values_list)

fd_indices = np.linspace(0, 4199, 128, dtype=int)

bsdr_indices.insert(0, fd_indices)
ad_indices.insert(0, fd_indices)

fig, axes = plt.subplots(3, 2, figsize=(8, 5))

ds = DSManager(name="min_lucas", shuffle=False)
original_data = ds.data[:, 0:-1]

row = original_data[1]
X = row

min_val = np.min(X)
max_val = np.max(X)

for epoch in range(3):
    axes[epoch,0].set_ylim([min_val, max_val])
    axes[epoch,1].set_ylim([min_val, max_val])

    bsdr_i = bsdr_indices[epoch]
    ad_i = ad_indices[epoch]

    h_bsdr = X[bsdr_i]
    h_ad = X[ad_i]

    axes[epoch,0].plot(h_bsdr, color="orange", label="BSDR")
    axes[epoch,0].set_xlabel("Index", fontsize=12)
    axes[epoch,0].set_ylabel("Value", fontsize=12)

    axes[epoch,1].plot(h_ad, color="blue", label="Adaptive-downsampling")
    axes[epoch,1].set_xlabel("Index", fontsize=12)
    axes[epoch,1].set_ylabel("Value", fontsize=12)

axes[-1,0].text(0.5, -0.8, "(a) BSDR", fontsize=14, ha="center", transform=axes[-1,0].transAxes)
axes[-1,1].text(1.7, -0.8, "(b) Adaptive-downsampling", fontsize=14, ha="center", transform=axes[-1,0].transAxes)


plt.tight_layout()
plt.savefig("low_rep_3_march_ba.png", dpi=600)
plt.clf()
plt.cla()
