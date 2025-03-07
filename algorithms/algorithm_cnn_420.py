from algorithm import Algorithm
import torch
import torch.nn as nn
from my_env import TEST
import torch.utils.data as data


class Algorithm_cnn_420(Algorithm):
    def __init__(self, dataset, train_x, train_y, test_x, test_y, target_size, fold, scaler_y, mode, train_size, reporter, verbose):
        super().__init__(dataset, train_x, train_y, test_x, test_y, target_size, fold, scaler_y, mode, train_size, reporter, verbose)

        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        torch.backends.cudnn.deterministic = True

        indices = torch.linspace(0, 4199, self.target_size, dtype=torch.int)

        self.train_x = torch.tensor(train_x, dtype=torch.float32).to(self.device)
        self.train_x = self.train_x[indices]
        self.train_y = torch.tensor(train_y, dtype=torch.float32).to(self.device)
        self.test_x = torch.tensor(test_x, dtype=torch.float32).to(self.device)
        self.test_x = self.test_x[indices]
        self.test_y = torch.tensor(test_y, dtype=torch.float32).to(self.device)

        self.criterion = torch.nn.MSELoss()
        self.class_size = 1
        self.lr = 0.001
        self.total_epoch = 2

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

        self.train_x = self.train_x.reshape(self.train_x.shape[0],1,self.train_x.shape[1])
        self.test_x = self.test_x.reshape(self.test_x.shape[0],1,self.test_x.shape[1])

        self.train_dataset = data.TensorDataset(self.train_x, self.train_y)
        self.test_dataset = data.TensorDataset(self.test_x, self.test_y)

        self.train_dataloader = data.DataLoader(self.train_dataset, batch_size=1000, shuffle=True)
        self.test_dataloader = data.DataLoader(self.test_dataset, batch_size=1000, shuffle=True)

        self.reporter.create_epoch_report(self.get_name(), self.dataset.name, self.target_size, self.scaler_y, self.mode, self.train_size, self.fold)

    def _fit(self):
        self.ann.train()
        optimizer = torch.optim.Adam(self.ann.parameters(), lr=self.lr, weight_decay=self.lr/10)

        for epoch in range(self.total_epoch):
            for index, (batch_x, batch_y) in enumerate(self.train_dataloader):
                optimizer.zero_grad()
                y_hat = self.ann(batch_x)
                y_hat = y_hat.reshape(-1)
                loss = self.criterion(y_hat, batch_y)
                loss.backward()
                optimizer.step()
                print(epoch, index, loss.item())
        return self

    def get_indices(self):
        return list(range(4200))

    def get_num_params(self):
        return sum(p.numel() for p in self.ann.parameters() if p.requires_grad)

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


