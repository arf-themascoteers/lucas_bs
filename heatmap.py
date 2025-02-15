import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

np.random.seed(42)
target_sizes = [10, 20, 30, 40, 50]
train_sizes = ["Train-10%", "Train-30%", "Train-50%", "Train-70%", "Train-90%"]

dummy_data = np.random.uniform(0.5, 1.0, size=(len(target_sizes), len(train_sizes)))
df_dummy = pd.DataFrame(dummy_data, index=target_sizes, columns=train_sizes)

plt.figure(figsize=(10, 6))
sns.heatmap(df_dummy, annot=True, cmap="viridis", linewidths=0.5, fmt=".2f")

plt.xlabel("Training Split")
plt.ylabel("Target Size")
plt.title("R² Values Across Training Splits and Target Sizes")

plt.show()
