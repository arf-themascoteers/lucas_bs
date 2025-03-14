import pandas as pd


df = pd.read_csv("imp_res/all_bands_m.csv")

df.loc[df['algorithm'] == 'bsdrcnn_r_4200_2', 'execution_time'] *= (1000/60)

df.to_csv("imp_res/all_bands_m2.csv", index=False)
