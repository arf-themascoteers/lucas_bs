import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# soc_data = pd.read_csv("data/lucas.csv").to_numpy()[:,-1]
# pd.DataFrame(data=soc_data, columns=["soc"]).to_csv("soc.csv", index=False)
soc_data = pd.read_csv("soc.csv").to_numpy()
plt.figure(figsize=(6, 4))
plt.hist(soc_data, bins=100, density=True, alpha=0.7, color="green")
plt.xlabel('SOC')
plt.ylabel('Density')
plt.title('SOC Distribution')
plt.savefig('soc_dist.png')
plt.show()
