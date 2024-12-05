from algorithm import Algorithm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class SAE(nn.Module):
    def __init__(self, input_size, target_size):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = input_size
        self.target_size = target_size

        self.encoder = nn.Sequential(
            nn.Linear(input_size, 2000),
            nn.Linear(2000, 1000),
            nn.Linear(1000, target_size),
        )

        self.decoder = nn.Sequential(
            nn.Linear(target_size, 1000),
            nn.Linear(1000, 2000),
            nn.Linear(2000, input_size)
        )

    def forward(self, X):
        hidden = self.encoder(X)
        output = self.decoder(hidden)
        return output


class ANN(nn.Module):
    def __init__(self, target_size):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_size = target_size
        last_layer_input = 640
        self.cnn = nn.Sequential(
            nn.Conv1d(1,32,kernel_size=8, stride=1, padding=0),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4, padding=0),

            nn.Conv1d(32, 64, kernel_size=8, stride=1, padding=0),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4, padding=0),

            nn.Conv1d(64, 128, kernel_size=8, stride=1, padding=0),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(kernel_size=4, stride=4, padding=0),

            nn.Flatten(start_dim=1),
            nn.Linear(last_layer_input,1)
        )
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        #print("Number of learnable parameters:", num_params)

    def forward(self, X):
        X = X.reshape(X.shape[0], 1, X.shape[1])
        X = self.cnn(X)
        X = X.reshape(-1)
        return X


class Algorithm_asa(Algorithm):
    def __init__(self, dataset, train_x, train_y, test_x, test_y, target_size, fold, scaler_y, mode, reporter, verbose):
        super().__init__(dataset, train_x, train_y, test_x, test_y, target_size, fold, scaler_y, mode, reporter, verbose)

        if self.mode in ["static", "semi"]:
            raise Exception("Unsupported mode")

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
        self.total_epoch = 400

        self.sae = SAE(self.train_x.shape[1],self.target_size)
        self.sae.to(self.device)
        self.criterion_sae = torch.nn.MSELoss(reduction='sum')
        self.criterion_ann = torch.nn.MSELoss(reduction='sum')

        self.ann = ANN(self.target_size)
        self.ann.to(self.device)
        self.original_feature_size = self.train_x.shape[1]


        self.reporter.create_epoch_report(self.get_name(), self.dataset.name, self.target_size, self.fold)

    def _fit(self):
        self.sae.train()
        self.ann.train()
        optimizer_sae = torch.optim.Adam(self.sae.parameters(), lr=self.lr, weight_decay=self.lr/10)
        optimizer_ann = torch.optim.Adam(self.ann.parameters(), lr=self.lr, weight_decay=self.lr/10)

        dataset = TensorDataset(self.train_x)
        dataloader = DataLoader(dataset, batch_size=20, shuffle=True)

        for epoch in range(self.total_epoch):
            for batch in dataloader:
                batch_x = batch[0]
                optimizer_sae.zero_grad()
                x_hat = self.sae(batch_x)
                loss = self.criterion_sae(x_hat, batch_x)
                loss.backward()
                optimizer_sae.step()
            if self.verbose and epoch % 10 == 0:
                print("AE",loss.item())

        for param in self.sae.encoder.parameters():
            param.requires_grad = False

        dataset = TensorDataset(self.train_x, self.train_y)
        dataloader = DataLoader(dataset, batch_size=20, shuffle=True)

        for epoch in range(self.total_epoch):
            for batch in dataloader:
                batch_x, batch_y = batch
                optimizer_ann.zero_grad()
                hidden = self.sae.encoder(batch_x)
                y_hat = self.ann(hidden)
                loss = self.criterion_ann(y_hat, batch_y)
                loss.backward()
                optimizer_ann.step()
            if self.verbose and epoch % 10 == 0:
                print("CNN",loss.item())

        torch.save(self.ann.state_dict(), 'ann.pth')
        torch.save(self.sae.state_dict(), 'sae.pth')
        return self

    def get_indices(self):
        return list(range(500))

    def predict_train(self):
        dataset = TensorDataset(self.train_x)
        dataloader = DataLoader(dataset, batch_size=20)
        outputs = []
        for batch in dataloader:
            batch_x = batch[0]
            hidden = self.sae.encoder(batch_x)
            batch_output = self.ann(hidden)
            outputs.append(batch_output)
        return torch.cat(outputs, dim=0)

    def predict_test(self):
        dataset = TensorDataset(self.test_x)
        dataloader = DataLoader(dataset, batch_size=20)
        outputs = []
        for batch in dataloader:
            batch_x = batch[0]
            hidden = self.sae.encoder(batch_x)
            batch_output = self.ann(hidden)
            outputs.append(batch_output)
        return torch.cat(outputs, dim=0)