import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("data/min_lucas.csv").to_numpy()

signal = df[0, 0:-1]
arr = np.linspace(0, 4199, 42, dtype=int)
downsampled = signal[arr]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(signal)
axes[0].set_title("Original Signal")

axes[1].plot(downsampled)
axes[1].set_title("Downsampled Signal")

plt.tight_layout()
plt.show()

