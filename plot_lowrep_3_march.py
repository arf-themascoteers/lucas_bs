import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from ds_manager import DSManager
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14
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

labels = ["(a)",
          "(b)",
          "(c)",
          "(d)",
          "(e)"]
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

    min_val = np.min(X)-0.01
    max_val = np.max(X)+0.01

    if p in [0, 1, 3, 4]:
        ax.set_ylim([min_val, max_val])

    if p == 0:
        ax.plot(X, color="green")
        ax.set_xlabel("Band")
        ax.set_ylabel("Reflectance")
        ax.set_title("Original data", fontsize=14)

    if p == 1:
        ax.plot(h_fd, color="purple")
        ax.set_xlabel("Band")
        ax.set_ylabel("Reflectance")
        ax.set_title("Fixed-interval downsampling", fontsize=14)

    if p == 2:
        ax.plot(h_sae, color="red", label="SAE")
        ax.set_xlabel("Index")
        ax.set_ylabel("Value")
        ax.set_title("SAE", fontsize=14)

    if p == 3:
        ax.plot(h_bsdr, color="orange")
        ax.set_xlabel("Band")
        ax.set_ylabel("Reflectance")
        ax.set_title("BSDR", fontsize=14)

    if p == 4:
        ax.plot(h_ad, color="blue")
        ax.set_xlabel("Band")
        ax.set_ylabel("Reflectance")
        ax.set_title("Adaptive downsampling", fontsize=14)

    ax.text(0.5, -0.6, labels[p], fontsize=16, ha="center",
        transform=ax.transAxes, fontname="Times New Roman")

axes[5].set_visible(False)
plt.subplots_adjust(left=0.1, right=0.95, top=0.90, bottom=0.1,
                    hspace=1, wspace=0.4)

plt.savefig("low_rep_3_march.png", dpi=600, pad_inches=0)
plt.show()
