import pandas as pd

df = pd.read_csv("back/check1/summary.csv")

df = df[(df["algorithm"] == "bsdrcnn") & (df["mode"] == "dyn")]
df.to_csv("adcnn.csv", index=False)
print(df)