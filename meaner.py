import pandas as pd
from ds_manager import DSManager
import numpy as np


ds = DSManager(name="lucas",folds=1,shuffle=False)
data = ds.data[:,0:-1]
data = ds.scaler_X.inverse_transform(data)
data = np.mean(data,axis=0,keepdims=True)
cols = [str(i) for i in range(data.shape[1])]
df = pd.DataFrame(data=data, columns=cols)
df.to_csv("mean.csv", index=False)