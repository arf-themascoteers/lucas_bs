import final_selected_bands as fsb
from ds_manager import DSManager
import numpy as np
from sklearn.metrics.pairwise import cosine_distances

ds = DSManager(name="lucas", shuffle=False)
data = ds.data[:, 0:-1]
bands = fsb.selected_bands_as_list

f_bands = fsb.fd_bands_as_list
b_bands = fsb.bsdr_bands_as_list


def mean_pairwise_dissimilarity(X, indices):
    D = cosine_distances(X[:, indices].T)
    return np.sum(D) / (D.shape[0] * (D.shape[0] - 1))


for k, v in bands.items():
    ad = mean_pairwise_dissimilarity(data, v)
    fd = mean_pairwise_dissimilarity(data, f_bands[k])
    bd = mean_pairwise_dissimilarity(data, b_bands[k])
    print(k)
    print(fd)
    print(bd)
    print(ad)
    print("------")
