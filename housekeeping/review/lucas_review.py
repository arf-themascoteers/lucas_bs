import pandas as pd


df = pd.read_csv("../../data/lucas_asa.csv")
df = df[df['oc'] > 0]
print(df["oc"].min())
print(df["oc"].max())
print(df["oc"].mean())
print(df["oc"].std())
cv = (df["oc"].std(ddof=1) / df["oc"].mean()) * 100
print(cv)
