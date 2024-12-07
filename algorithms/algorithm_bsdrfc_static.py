from algorithm import Algorithm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class ANN(nn.Module):
    def __init__(self, target_size, mode):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_size = target_size
        self.mode = mode

        self.indices = torch.linspace(0, target_size - 1, target_size).to(self.device)
        if self.mode in ["static", "semi"]:
            self.indices.requires_grad = False
        self.fc = nn.Sequential(
            nn.Linear(self.target_size, 40),
            nn.BatchNorm1d(40),
            nn.LeakyReLU(),
            nn.Linear(40,1)
        )
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        #print("Number of learnable parameters:", num_params)

    @staticmethod
    def inverse_sigmoid_torch(x):
        return -torch.log(1.0 / x - 1.0)

    def forward(self, X):
        soc_hat = self.fc(X)
        soc_hat = soc_hat.reshape(-1)
        return soc_hat

    def get_indices(self):
        return torch.sigmoid(self.indices)


class Algorithm_bsdrfc_static(Algorithm):
    def __init__(self, dataset, train_x, train_y, test_x, test_y, target_size, fold, scaler_y, mode, reporter, verbose):
        super().__init__(dataset, train_x, train_y, test_x, test_y, target_size, fold, scaler_y, mode, reporter, verbose)

        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        torch.backends.cudnn.deterministic = True

        self.train_x = torch.tensor(train_x, dtype=torch.float32).to(self.device)
        self.train_y = torch.tensor(train_y, dtype=torch.float32).to(self.device)
        self.test_x = torch.tensor(test_x, dtype=torch.float32).to(self.device)
        self.test_y = torch.tensor(test_y, dtype=torch.float32).to(self.device)

        self.criterion = torch.nn.MSELoss()
        self.class_size = 1
        self.lr = 0.001
        self.total_epoch = 30

        self.ann = ANN(self.target_size, mode)
        self.ann.to(self.device)
        self.original_feature_size = self.train_x.shape[1]

        self.reporter.create_epoch_report(self.get_name(), self.dataset.name, self.target_size, self.scaler_y, self.mode, self.fold)

    def _fit(self):
        self.ann.train()
        self.write_columns()
        optimizer = torch.optim.Adam(self.ann.parameters(), lr=self.lr, weight_decay=self.lr/10)

        dataset = TensorDataset(self.train_x, self.train_y)
        dataloader = DataLoader(dataset, batch_size=5, shuffle=True)

        for epoch in range(self.total_epoch):
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                y_hat = self.ann(batch_x)
                loss = self.criterion(y_hat, batch_y)
                loss.backward()
                optimizer.step()
            self.report(epoch)
        return self

    def predict_train(self):
        dataset = TensorDataset(self.train_x, self.train_y)
        dataloader = DataLoader(dataset, batch_size=50, shuffle=True)
        y_hats = []
        for epoch in range(self.total_epoch):
            for batch_x, batch_y in dataloader:
                y_hat = self.ann(batch_x)
                y_hats.append(y_hat)
        return torch.cat(y_hats, dim=0)


    def predict_test(self):
        dataset = TensorDataset(self.test_x, self.test_y)
        dataloader = DataLoader(dataset, batch_size=50, shuffle=True)
        y_hats = []
        for epoch in range(self.total_epoch):
            for batch_x, batch_y in dataloader:
                y_hat = self.ann(batch_x)
                y_hats.append(y_hat)
        return torch.cat(y_hats, dim=0)


    def write_columns(self):
        if not self.verbose:
            return
        columns = ["epoch","r2","rmse","rpd","rpiq","train_r2","train_rmse","train_rpd","train_rpiq"] + [f"band_{index+1}" for index in range(self.target_size)]
        print("".join([str(i).ljust(20) for i in columns]))

    def report(self, epoch):
        if not self.verbose:
            return
        if epoch%10 != 0:
            return

        bands = self.get_indices()

        train_y_hat = self.predict_train()
        test_y_hat = self.predict_test()

        r2, rmse, rpd, rpiq = self.calculate_metrics(self.test_y, test_y_hat)
        train_r2, train_rmse, train_rpd, train_rpiq = self.calculate_metrics(self.train_y, train_y_hat)

        self.reporter.report_epoch_bsdr(epoch, r2, rpd, rpiq, rmse, train_r2, train_rmse, train_rpd, train_rpiq, bands)
        cells = [epoch, r2, rmse, rpd, rpiq, train_r2, train_rmse, train_rpd, train_rpiq] + bands
        cells = [round(item, 5) if isinstance(item, float) else item for item in cells]
        print("".join([str(i).ljust(20) for i in cells]))

    def get_indices(self):
        indices = torch.round(self.ann.get_indices() * self.original_feature_size ).to(torch.int64).tolist()
        return list(dict.fromkeys(indices))
