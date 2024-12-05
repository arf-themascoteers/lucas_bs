import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


data = pd.read_csv("vecs.csv").to_numpy()
X_mean = np.mean(data[:,0:4200], axis=0)
X_hat_mean = np.mean(data[:,4700:], axis=0)
plt.plot(X_mean, color="green", label="Original data")
plt.plot(X_hat_mean, color="blue", linestyle='--', dashes=(1, 3), label="Reconstructed data")


plt.xlabel("Band", fontsize=15)
plt.ylabel("Reflectance (normalized)", fontsize=15)
plt.legend(loc="upper center", fontsize=12, ncols=2, bbox_to_anchor=(0.5, 1.15))
plt.savefig("disc2.png")
plt.show()


