import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
df = pd.read_csv("data/min_lucas.csv").to_numpy()

signal = df[0, 0:-1]
arr = np.linspace(0, 4199, 42, dtype=int)
downsampled = signal[arr]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(signal)
axes[0].set_title("(a) Original Signal")

axes[1].plot(downsampled)
axes[1].set_title("(b) Downsampled Signal (ordered)")

df = pd.read_csv("for_plots/sc/r_wo_order.csv")
last_row = df.iloc[-1]
values_list = last_row.loc['band_1':].dropna().tolist()
values_list = [int(i) for i in values_list]
print(len(values_list))
downsampled_unordered = signal[values_list]

axes[2].plot(downsampled_unordered)
axes[2].set_title("(b) Downsampled Signal (unordered)")
print(len(downsampled_unordered))
print(values_list)
plt.tight_layout()
plt.show()


