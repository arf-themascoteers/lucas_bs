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
        self.indices = torch.linspace(0,target_size-1,target_size).to(self.device)

        self.cnn = self.get_cnn(self.target_size)
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        #print("Number of learnable parameters:", num_params)

    def get_cnn(self, target_size):
        if target_size == 8:
            return nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=4, stride=1, padding=0),
                nn.BatchNorm1d(32),
                nn.LeakyReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
                nn.Flatten(start_dim=1),
                nn.Linear(64, 1)
            )
        if target_size == 16:
            return nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=4, stride=1, padding=0),
                nn.BatchNorm1d(32),
                nn.LeakyReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
                nn.Conv1d(32, 64, kernel_size=2, stride=1, padding=0),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
                nn.Flatten(start_dim=1),
                nn.Linear(128, 1)
            )
        if target_size == 32:
            return nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=8, stride=1, padding=0),
                nn.BatchNorm1d(32),
                nn.LeakyReLU(),
                nn.MaxPool1d(kernel_size=4, stride=4, padding=0),
                nn.Conv1d(32, 64, kernel_size=2, stride=1, padding=0),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
                nn.Flatten(start_dim=1),
                nn.Linear(128, 1)
            )

        if target_size == 64:
            return nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=8, stride=1, padding=0),
                nn.BatchNorm1d(32),
                nn.LeakyReLU(),
                nn.MaxPool1d(kernel_size=4, stride=4, padding=0),
                nn.Conv1d(32, 64, kernel_size=4, stride=1, padding=0),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(),
                nn.MaxPool1d(kernel_size=4, stride=4, padding=0),
                nn.Flatten(start_dim=1),
                nn.Linear(128, 1)
            )
        if target_size == 128:
            return nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=16, stride=1, padding=0),
                nn.BatchNorm1d(32),
                nn.LeakyReLU(),
                nn.MaxPool1d(kernel_size=8, stride=8, padding=0),
                nn.Conv1d(32, 64, kernel_size=8, stride=1, padding=0),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(),
                nn.MaxPool1d(kernel_size=4, stride=4, padding=0),
                nn.Flatten(start_dim=1),
                nn.Linear(64, 1)
            )
        if target_size == 256:
            return nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=16, stride=1, padding=0),
                nn.BatchNorm1d(32),
                nn.LeakyReLU(),
                nn.MaxPool1d(kernel_size=8, stride=8, padding=0),
                nn.Conv1d(32, 64, kernel_size=8, stride=1, padding=0),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(),
                nn.MaxPool1d(kernel_size=8, stride=8, padding=0),
                nn.Flatten(start_dim=1),
                nn.Linear(128, 1)
            )
        if target_size == 500:
            return nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=16, stride=1, padding=0),
                nn.BatchNorm1d(32),
                nn.LeakyReLU(),
                nn.MaxPool1d(kernel_size=8, stride=8, padding=0),
                nn.Conv1d(32, 64, kernel_size=8, stride=1, padding=0),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(),
                nn.MaxPool1d(kernel_size=8, stride=8, padding=0),
                nn.Conv1d(64, 128, kernel_size=4, stride=1, padding=0),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
                nn.Flatten(start_dim=1),
                nn.Linear(128, 1)
            )
        if target_size == 512:
            return nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=16, stride=1, padding=0),
                nn.BatchNorm1d(32),
                nn.LeakyReLU(),
                nn.MaxPool1d(kernel_size=8, stride=8, padding=0),
                nn.Conv1d(32, 64, kernel_size=8, stride=1, padding=0),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(),
                nn.MaxPool1d(kernel_size=8, stride=8, padding=0),
                nn.Conv1d(64, 128, kernel_size=4, stride=1, padding=0),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
                nn.Flatten(start_dim=1),
                nn.Linear(128, 1)
            )
        if target_size == 1000:
            return nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=16, stride=1, padding=0),
                nn.BatchNorm1d(32),
                nn.LeakyReLU(),
                nn.MaxPool1d(kernel_size=8, stride=8, padding=0),
                nn.Conv1d(32, 64, kernel_size=8, stride=1, padding=0),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(),
                nn.MaxPool1d(kernel_size=8, stride=8, padding=0),
                nn.Conv1d(64, 128, kernel_size=4, stride=1, padding=0),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(),
                nn.MaxPool1d(kernel_size=4, stride=4, padding=0),
                nn.Flatten(start_dim=1),
                nn.Linear(256, 1)
            )
        if target_size == 1024:
            return nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=16, stride=1, padding=0),
                nn.BatchNorm1d(32),
                nn.LeakyReLU(),
                nn.MaxPool1d(kernel_size=8, stride=8, padding=0),
                nn.Conv1d(32, 64, kernel_size=8, stride=1, padding=0),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(),
                nn.MaxPool1d(kernel_size=8, stride=8, padding=0),
                nn.Conv1d(64, 128, kernel_size=4, stride=1, padding=0),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(),
                nn.MaxPool1d(kernel_size=4, stride=4, padding=0),
                nn.Flatten(start_dim=1),
                nn.Linear(256, 1)
            )
        if target_size == 2000:
            return nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=32, stride=1, padding=0),
                nn.BatchNorm1d(32),
                nn.LeakyReLU(),
                nn.MaxPool1d(kernel_size=16, stride=16, padding=0),
                nn.Conv1d(32, 64, kernel_size=16, stride=1, padding=0),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(),
                nn.MaxPool1d(kernel_size=8, stride=8, padding=0),
                nn.Conv1d(64, 128, kernel_size=4, stride=1, padding=0),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(),
                nn.MaxPool1d(kernel_size=4, stride=4, padding=0),
                nn.Flatten(start_dim=1),
                nn.Linear(256, 1)
            )
        if target_size == 2048:
            return nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=32, stride=1, padding=0),
                nn.BatchNorm1d(32),
                nn.LeakyReLU(),
                nn.MaxPool1d(kernel_size=16, stride=16, padding=0),
                nn.Conv1d(32, 64, kernel_size=16, stride=1, padding=0),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(),
                nn.MaxPool1d(kernel_size=8, stride=8, padding=0),
                nn.Conv1d(64, 128, kernel_size=4, stride=1, padding=0),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(),
                nn.MaxPool1d(kernel_size=4, stride=4, padding=0),
                nn.Flatten(start_dim=1),
                nn.Linear(256, 1)
            )
        if target_size == 4200:
            return nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=32, stride=16, padding=0),
                nn.BatchNorm1d(32),
                nn.LeakyReLU(),
                nn.MaxPool1d(kernel_size=16, stride=16, padding=0),
                nn.Conv1d(32, 64, kernel_size=8, stride=1, padding=0),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(),
                nn.MaxPool1d(kernel_size=4, stride=4, padding=0),
                nn.LeakyReLU(),
                nn.Flatten(start_dim=1),
                nn.Linear(128, 1)
            )

        return None

    @staticmethod
    def inverse_sigmoid_torch(x):
        return -torch.log(1.0 / x - 1.0)

    def forward(self, X):
        outputs = X.reshape(X.shape[0], 1, X.shape[1])
        soc_hat = self.cnn(outputs)
        soc_hat = soc_hat.reshape(-1)
        return soc_hat

    def get_indices(self):
        return torch.sigmoid(self.indices)


class Algorithm_bsdrcnn_static(Algorithm):
    def __init__(self, dataset, train_x, train_y, test_x, test_y, target_size, fold, scaler_y, mode, train_split, reporter, verbose):
        super().__init__(dataset, train_x, train_y, test_x, test_y, target_size, fold, scaler_y, mode, train_split, reporter, verbose)

        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        torch.backends.cudnn.deterministic = True

        self.train_x = torch.tensor(train_x, dtype=torch.float32)
        self.train_y = torch.tensor(train_y, dtype=torch.float32)
        self.test_x = torch.tensor(test_x, dtype=torch.float32)
        self.test_y = torch.tensor(test_y, dtype=torch.float32)

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
        dataloader = DataLoader(dataset, batch_size=50, shuffle=True)

        for epoch in range(self.total_epoch):
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                y_hat = self.ann(batch_x)
                loss = self.criterion(y_hat, batch_y)
                loss.backward()
                optimizer.step()
                del batch_x, batch_y, y_hat, loss
                print("batch")
            self.report(epoch)
        return self

    def predict_train(self):
        dataset = TensorDataset(self.train_x, self.train_y)
        dataloader = DataLoader(dataset, batch_size=50, shuffle=True)
        y_hats = []
        for epoch in range(self.total_epoch):
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                y_hat = self.ann(batch_x)
                y_hats.append(y_hat)
        return torch.cat(y_hats, dim=0)


    def predict_test(self):
        dataset = TensorDataset(self.test_x, self.test_y)
        dataloader = DataLoader(dataset, batch_size=50, shuffle=True)
        y_hats = []
        for epoch in range(self.total_epoch):
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
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
