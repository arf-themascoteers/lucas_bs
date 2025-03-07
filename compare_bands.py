import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("imp_res/res.csv")
last_row = df.iloc[-1]
values_list = last_row.loc['band_1':].dropna().tolist()
values_list = [int(i) for i in values_list]
print(len(values_list))
downsampled_unordered = signal[values_list]