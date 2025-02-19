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

print(", ".join(map(str, arr)))
np.random.shuffle(arr)
downsampled2 = signal[arr]

axes[2].plot(downsampled2)
axes[2].set_title("(c) Downsampled Signal (unordered)")

print(", ".join(map(str, arr)))

plt.tight_layout()
plt.show()

