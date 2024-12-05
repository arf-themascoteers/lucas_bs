import pandas as pd
import matplotlib.pyplot as plt

fig, axes = plt.subplots(5, 2, figsize=(10, 15))
data = pd.read_csv("vecs.csv").to_numpy()
axes = axes.flatten()

for idx, ax in enumerate(axes):
    row = data[idx]
    X = row[0:4200]
    h = row[4200:4700]
    X_hat = row[4700:]
    ax.plot(X_hat)
    #ax.plot(X_hat)
    ax.set_title(f"Plot {idx + 1}")

plt.tight_layout()
plt.show()



