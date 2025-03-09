import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from ds_manager import DSManager

bsdr_indices = []
df = pd.read_csv("backup_results/2_compo/bsdrfc_r_norder_lucas_128_robust_dyn_0.75_0.csv")
for i in range(3):
    last_row = df.iloc[i+1]
    values_list = last_row.loc['band_1':].dropna().tolist()
    values_list = [int(i) for i in values_list]
    values_list = np.array(values_list)
    bsdr_indices.append(values_list)

bsdr_indices = bsdr_indices[1]

ad_indices = []
df = pd.read_csv("backup_results/2_compo/bsdrcnn_r_lucas_128_robust_dyn_0.75_0.csv")
for i in range(3):
    last_row = df.iloc[i+1]
    values_list = last_row.loc['band_1':].dropna().tolist()
    values_list = [int(i) for i in values_list]
    ad_indices.append(values_list)

fig, axes = plt.subplots(3, 2, figsize=(7, 7))
axes = axes.ravel()

ad_indices = ad_indices[1]

labels = ["(a) Original data",
          "(b) Fixed-interval downsampling",
          "(c) SAE",
          "(d) BSDR",
          "(e)Adaptive-downsampling"]
for p in range(5):
    sae_data = pd.read_csv("v2.csv").to_numpy()
    ds = DSManager(name="min_lucas", shuffle=False)
    original_data = ds.data[:, 0:-1]

    fd_indices = np.linspace(0, 4199, 128, dtype=int)
    fd_data = original_data[:, fd_indices]

    print(len(bsdr_indices), len(ad_indices))
    bsdr_data = original_data[:, bsdr_indices]

    ad_indices = [int(i) for i in ad_indices]
    ad_data = original_data[:, ad_indices]

    sindices = [1, 3, 9]

    i = 0
    sindex = sindices[i]
    row = original_data[sindex]
    X = row
    h_fd = fd_data[sindex]
    h_sae = sae_data[i][4200:4700]
    h_bsdr = bsdr_data[sindex]
    h_ad = ad_data[sindex]

    ax = axes[p]

    min_val = np.min(X)
    max_val = np.max(X)

    if p in [0, 1, 3, 4]:
        ax.set_ylim([min_val, max_val])

    if p == 0:
        ax.plot(X, color="green", label="Original data")
        ax.set_xlabel("Band", fontsize=12)
        ax.set_ylabel("Reflectance", fontsize=12)

    if p == 1:
        ax.plot(h_fd, color="purple", label="Fixed-interval downsampling")
        ax.set_xlabel("Index", fontsize=12)
        ax.set_ylabel("Value", fontsize=12)

    if p == 2:
        ax.plot(h_sae, color="red", label="SAE")
        ax.set_xlabel("Index", fontsize=12)
        ax.set_ylabel("Value", fontsize=12)

    if p == 3:
        ax.plot(h_bsdr, color="orange", label="BSDR")
        ax.set_xlabel("Index", fontsize=12)
        ax.set_ylabel("Value", fontsize=12)

    if p == 4:
        ax.plot(h_ad, color="blue", label="Adaptive-downsampling")
        ax.set_xlabel("Index", fontsize=12)
        ax.set_ylabel("Value", fontsize=12)

    ax.text(0.5, -0.5, labels[p], fontsize=14, ha="center", transform=ax.transAxes)

axes[5].set_visible(False)
plt.tight_layout()
plt.savefig("low_rep_3_march.png", dpi=600)
plt.clf()
plt.cla()
