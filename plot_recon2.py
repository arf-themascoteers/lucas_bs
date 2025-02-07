import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from ds_manager import DSManager


ds = DSManager(name="min_lucas")
data = pd.read_csv("vecs.csv").to_numpy()
data[:, 0:4200] = ds.scaler_X.inverse_transform(data[:, 0:4200])
data[:, 4700:] = ds.scaler_X.inverse_transform(data[:, 4700:])
X_mean = np.mean(data[:,0:4200], axis=0)
X_hat_mean = np.mean(data[:,4700:], axis=0)
plt.plot(X_mean, color="green", label="Mean original spectra")
plt.plot(X_hat_mean, color="red", linestyle='--', dashes=(1, 3), label="Mean reconstructed spectra")


plt.xlabel("Band", fontsize=15)
plt.ylabel("Reflectance", fontsize=15)
plt.legend(loc="upper center", fontsize=12, ncols=2, bbox_to_anchor=(0.5, 1.15))
plt.savefig("means.png")
plt.show()


