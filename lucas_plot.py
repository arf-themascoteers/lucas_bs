import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv("data/lucas_crop.csv").to_numpy()[:,0:-1]
data = np.mean(data, axis=0)
plt.plot(data)
plt.show()