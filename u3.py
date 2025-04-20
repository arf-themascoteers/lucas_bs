import final_selected_bands as fsb
from ds_manager import DSManager
import numpy as np


ds = DSManager(name="lucas",shuffle=False)
data = ds.data[:,0:-1]
bands = fsb.selected_bands_as_list

f_bands = fsb.fd_bands_as_list
b_bands = fsb.bsdr_bands_as_list

for k,v in bands.items():
    ad = np.mean(np.var(data[:, v], axis=0))
    fd = np.mean(np.var(data[:, f_bands[k]], axis=0))
    bd = np.mean(np.var(data[:, b_bands[k]], axis=0))
    print(k)
    print(fd)
    print(bd)
    print(ad)
    print("------")