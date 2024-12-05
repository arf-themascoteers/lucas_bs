import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import mahalanobis
from scipy.signal import savgol_filter
from kennard_stone import train_test_split as ks_split

df = pd.read_csv('../../data/lucas.csv')
df = df.sample(frac=0.01).reset_index(drop=True)
df.to_csv('../../data/min_lucas.csv')