from algorithm import Algorithm
import torch
import torch.nn as nn
from my_env import TEST
import torch.utils.data as data


class Algorithm_bsdrcnn_r_4200_2(Algorithm):
    def __init__(self, dataset, train_x, train_y, test_x, test_y, target_size, fold, scaler_y, mode, train_size, reporter, verbose):
        super().__init__(dataset, train_x, train_y, test_x, test_y, target_size, fold, scaler_y, mode, train_size, reporter, verbose)

        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        torch.backends.cudnn.deterministic = True

        self.index_list = torch.linspace(0, 4199, steps=self.target_size, dtype=torch.int)

        self.train_x = torch.tensor(train_x, dtype=torch.float32).to(self.device)
        self.train_x = self.train_x[:,self.index_list]
        self.train_x = self.train_x.reshape(self.train_x.shape[0],1,self.train_x.shape[1])
        self.train_y = torch.tensor(train_y, dtype=torch.float32).to(self.device)

        self.test_x = torch.tensor(test_x, dtype=torch.float32).to(self.device)
        self.test_x = self.test_x[:, self.index_list]
        self.test_x = self.test_x.reshape(self.test_x.shape[0],1,self.test_x.shape[1])
        self.test_y = torch.tensor(test_y, dtype=torch.float32).to(self.device)

        self.train_dataset = data.TensorDataset(self.train_x, self.train_y)
        self.test_dataset = data.TensorDataset(self.test_x, self.test_y)

        self.train_dataloader = data.DataLoader(self.train_dataset, batch_size=1000, shuffle=False)
        self.test_dataloader = data.DataLoader(self.test_dataset, batch_size=1000, shuffle=False)

        self.criterion = torch.nn.MSELoss()
        self.class_size = 1
        self.lr = 0.001
        self.total_epoch = 60

        if TEST:
            self.total_epoch = 1
            print(test_y.shape)

        self.ann = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=16, stride=1, padding=0),
                nn.BatchNorm1d(32),
                nn.LeakyReLU(),
                nn.MaxPool1d(kernel_size=16, stride=16, padding=0),
                nn.Conv1d(32, 64, kernel_size=8, stride=1, padding=0),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(),
                nn.MaxPool1d(kernel_size=16, stride=16, padding=0),
                nn.Conv1d(64, 128, kernel_size=4, stride=1, padding=0),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(),
                nn.MaxPool1d(kernel_size=8, stride=8, padding=0),
                nn.Flatten(start_dim=1),
                nn.Linear(128, 1)
            )
        self.ann.to(self.device)
        self.original_feature_size = self.train_x.shape[1]
        self.reporter.create_epoch_report(self.get_name(), self.dataset.name, self.target_size, self.scaler_y, self.mode, self.train_size, self.fold)

    def _fit(self):
        self.ann.train()
        self.write_columns()
        optimizer = torch.optim.Adam(self.ann.parameters(), lr=self.lr, weight_decay=self.lr/10)
        for epoch in range(self.total_epoch):
            for index, (batch_x, batch_y) in enumerate(self.train_dataloader):
                optimizer.zero_grad()
                y_hat = self.ann(batch_x)
                y_hat = y_hat.reshape(-1)
                loss = self.criterion(y_hat, batch_y)
                loss.backward()
                optimizer.step()
            if self.verbose:
                self.report(epoch)
        return self

    def predict_train(self):
        train_y_hat = []

        for index, (batch_x, batch_y) in enumerate(self.train_dataloader):
            y_hat = self.ann(batch_x)
            y_hat = y_hat.reshape(-1)
            train_y_hat.append(y_hat.detach().cpu())

        train_y_hat = torch.cat(train_y_hat, dim=0)
        return train_y_hat

    def predict_test(self):
        test_y_hat = []

        for index, (batch_x, batch_y) in enumerate(self.test_dataloader):
            y_hat = self.ann(batch_x)
            y_hat = y_hat.reshape(-1)
            test_y_hat.append(y_hat.detach().cpu())

        test_y_hat = torch.cat(test_y_hat, dim=0)
        return test_y_hat

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

        r2, rmse, rpd, rpiq, r2_o, rmse_o, rpd_o, rpiq_o \
            = self.calculate_metrics(self.test_y, test_y_hat)
        train_r2, train_rmse, train_rpd, train_rpiq, train_r2_o, train_rmse_o, train_rpd_o, train_rpiq_o \
            = self.calculate_metrics(self.train_y, train_y_hat)

        self.reporter.report_epoch_bsdr(epoch, r2, rpd, rpiq, rmse, train_r2, train_rmse, train_rpd, train_rpiq, bands)
        cells = [epoch, r2, rmse, rpd, rpiq, train_r2, train_rmse, train_rpd, train_rpiq] + bands
        cells = [round(item, 5) if isinstance(item, float) else item for item in cells]
        print("".join([str(i).ljust(20) for i in cells]))

    def get_indices(self):
        indices = self.index_list.tolist()
        indices = [int(i) for i in indices]
        return indices

    def get_num_params(self):
        return sum(p.numel() for p in self.ann.parameters() if p.requires_grad)
