import pandas as pd
from ds_manager import DSManager
import matplotlib.pyplot as plt


ds = DSManager(name="min_lucas",shuffle=False)
data = ds.data[:,0:-1]
#indices = [59,188,218,260,339,379,420,514,614,683,740,811,854,861,938,979,1057,1104,1196,1257,1339,1362,1445,1591,1594,1596,1806,1838,1869,1936,2009,2027,2152,2179,2214,2217,2329,2393,2519,2578,2638,2639,2713,2828,2833,2969,2982,3058,3107,3122,3229,3325,3479,3500,3515,3537,3663,3664,3752,3902,3908,3941,4021,4082]
indices = [28, 126, 142, 164, 172, 198, 231, 295, 310, 321, 414, 501, 503, 517, 522, 525, 601, 612, 660, 661, 662, 664, 683, 700, 723, 760, 852, 934, 945, 974, 1017, 1024, 1032, 1035, 1050, 1057, 1065, 1090, 1129, 1135, 1220, 1230, 1254, 1265, 1316, 1425, 1526, 1556, 1597, 1599, 1641, 1764, 1850, 1872, 1890, 1921, 1922, 1924, 1976, 1981, 1983, 1984, 1987, 2015, 2084, 2138, 2142, 2144, 2148, 2149, 2187, 2252, 2275, 2368, 2373, 2513, 2532, 2533, 2536, 2540, 2596, 2654, 2756, 2783, 2900, 2918, 2930, 2959, 2989, 3001, 3018, 3047, 3153, 3183, 3203, 3275, 3318, 3319, 3470, 3475, 3477, 3478, 3568, 3608, 3610, 3733, 3736, 3741, 3766, 3775, 3820, 3899, 3903, 3933, 3967, 3969, 3985, 4115, 4144]
print(len(indices))
indices = [int(i) for i in indices]
mdata = ds.data[:,indices]



fig, axes = plt.subplots(3, 2, figsize=(7, 7))
fig.subplots_adjust(hspace=0.5)


for i in range(3):
    row = data[i]
    X = row
    h = mdata[i]

    axes[i,0].set_ylim([0,1])
    axes[i,1].set_ylim([0,1])

    axes[i,0].plot(X, color="green", label="Original data")
    axes[i,0].set_title(f"Sample {i + 1}")
    axes[i,0].set_xlabel("Band")
    axes[i,0].set_ylabel("Reflectance")

    axes[i,1].plot(h, color="red", label="Compressed data")
    axes[i,1].set_title(f"Low-dimensional sample {i + 1}")
    axes[i,1].set_xlabel("Index")
    axes[i,1].set_ylabel("Value")


lines, labels = axes[0,1].get_legend_handles_labels()
#fig.legend(lines, labels, loc='upper center', ncol=2,bbox_to_anchor=(0.5,1))

#fig.subplots_adjust(top=0.9)
plt.tight_layout()
plt.savefig("low_rep_ad.png")
plt.show()
