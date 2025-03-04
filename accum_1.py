import pandas as pd

df1 = pd.read_csv("results/11_best/summary.csv")
df1 = df1[df1["train_size"] == 0.75]

df2 = pd.read_csv("results/9_bsdr_75/summary.csv")
df2 = df2[df2["train_size"] == 0.75]

df3 = pd.read_csv("results/7_static_75/summary.csv")
df3 = df3[df3["train_size"] == 0.75]

df4 = pd.read_csv("results/6_fc_75/summary.csv")
df4 = df4[df4["train_size"] == 0.75]

df_combined = pd.concat([df1, df2, df3, df4], ignore_index=True)
df_combined.to_csv("imp_res/ablation1.csv", index=False)