import torch
from ds_manager import DSManager
from algorithms.algorithm_asa import SAE
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ds = DSManager(name="min_lucas")
X = torch.tensor(ds.data[:,0:-1], dtype=torch.float32).to(device)

model = SAE(4200,500).to(device)
model.load_state_dict(torch.load("sae.pth", map_location=device, weights_only=True))
model.eval()

x_hat = model(X)
print(x_hat.shape)
h_hat = model.encoder(X)
print(h_hat.shape)

indices = [0,1,2,3,4,5,6,7,8,9]
X_short = X[indices]
h_short = h_hat[indices]
X_hat_short = x_hat[indices]

cols = [f"o_{i}" for i in range(X_short.shape[1])]
cols = cols + [f"h_{i}" for i in range(h_short.shape[1])]
cols = cols + [f"p_{i}" for i in range(X_hat_short.shape[1])]

data = torch.hstack((X_short, h_short, X_hat_short))
df = pd.DataFrame(data=data.detach().cpu().numpy(), columns=cols)
df.to_csv("vecs.csv", index=False)