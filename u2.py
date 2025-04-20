import final_selected_bands as fsb
from ds_manager import DSManager
import numpy as np
from sklearn.metrics import normalized_mutual_info_score


ds = DSManager(name="lucas",shuffle=False)
data = ds.data[:,0:-1]
bands = fsb.selected_bands_as_list

f_bands = fsb.fd_bands_as_list
b_bands = fsb.bsdr_bands_as_list

def mean_pairwise_mutual_info(X, indices):
    subset = X[:, indices]
    n = subset.shape[1]
    total = 0
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            x = np.digitize(subset[:, i], bins=np.histogram_bin_edges(subset[:, i], bins='auto'))
            y = np.digitize(subset[:, j], bins=np.histogram_bin_edges(subset[:, j], bins='auto'))
            total += normalized_mutual_info_score(x, y)
            count += 1
    return total / count if count > 0 else 0

for k, v in bands.items():
    ad = mean_pairwise_mutual_info(data, v)
    fd = mean_pairwise_mutual_info(data, f_bands[k])
    bd = mean_pairwise_mutual_info(data, b_bands[k])
    print(k)
    print(fd)
    print(bd)
    print(ad)
    print("------")