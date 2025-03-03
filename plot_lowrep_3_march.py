import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from ds_manager import DSManager

fig, axes = plt.subplots(3, 2, figsize=(7, 7))
axes = axes.ravel()

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

    bsdr_indices = [10, 168, 230, 34, 257, 311, 361, 447, 341, 354, 367, 357, 381, 418, 529, 409, 575, 657, 558, 747,
                    710, 785, 680, 738, 720, 878, 890, 812, 807, 963, 969, 956, 1041, 1009, 1049, 1188, 1096, 1326,
                    1198, 1303, 1337, 1362, 1378, 1459, 1371, 1420, 1437, 1481, 1428, 1592, 1588, 1679, 1766, 1701,
                    1819, 1872, 1799, 1826, 1942, 2039, 1944, 2024, 2087, 2158, 2149, 2093, 2064, 2326, 2211, 2303,
                    2435, 2363, 2532, 2308, 2652, 2497, 2653, 2655, 2569, 2590, 2504, 2745, 2622, 2608, 2659, 2704,
                    2767, 2813, 2966, 2971, 2985, 2997, 3044, 3043, 3066, 3159, 3150, 3207, 3198, 3470, 3474, 3467,
                    3362, 3412, 3360, 3513, 3503, 3506, 3496, 3561, 3655, 3549, 3676, 3703, 3705, 3692, 3710, 3732,
                    3910, 3926, 3914, 3909, 3913, 4110, 4086, 4067, 4179]
    bsdr_data = original_data[:, bsdr_indices]

    ad_indices = [28, 126, 142, 164, 172, 198, 231, 295, 310, 321, 414, 501, 503, 517, 522, 525, 601, 612, 660, 661, 662, 664, 683, 700, 723, 760, 852, 934, 945, 974, 1017, 1024, 1032, 1035, 1050, 1057, 1065, 1090, 1129, 1135, 1220, 1230, 1254, 1265, 1316, 1425, 1526, 1556, 1597, 1599, 1641, 1764, 1850, 1872, 1890, 1921, 1922, 1924, 1976, 1981, 1983, 1984, 1987, 2015, 2084, 2138, 2142, 2144, 2148, 2149, 2187, 2252, 2275, 2368, 2373, 2513, 2532, 2533, 2536, 2540, 2596, 2654, 2756, 2783, 2900, 2918, 2930, 2959, 2989, 3001, 3018, 3047, 3153, 3183, 3203, 3275, 3318, 3319, 3470, 3475, 3477, 3478, 3568, 3608, 3610, 3733, 3736, 3741, 3766, 3775, 3820, 3899, 3903, 3933, 3967, 3969, 3985, 4115, 4144]
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
        ax.plot(h_fd, color="orange", label="Fixed-interval downsampling")
        ax.set_xlabel("Index", fontsize=12)
        ax.set_ylabel("Value", fontsize=12)

    if p == 2:
        ax.plot(h_sae, color="red", label="SAE")
        ax.set_xlabel("Index", fontsize=12)
        ax.set_ylabel("Value", fontsize=12)

    if p == 3:
        ax.plot(h_bsdr, color="yellow", label="BSDR")
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
