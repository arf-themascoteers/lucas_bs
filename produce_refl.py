import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from ds_manager import DSManager
import csv



ds = DSManager(name="min_lucas", shuffle=False)
X = ds.data[:,0:-1]
X = np.mean(X, axis=0)
np.savetxt("X_mean_min.csv", X[None, :], delimiter=",")
