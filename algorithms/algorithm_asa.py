from algorithm import Algorithm
import torch
import torch.nn as nn


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
        return hidden, output


class ANN(nn.Module):
    def __init__(self, target_size):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_size = target_size
        last_layer_input = 684
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
        print("Number of learnable parameters:", num_params)

    @staticmethod
    def inverse_sigmoid_torch(x):
        return -torch.log(1.0 / x - 1.0)

    def forward(self, linterp):
        outputs = linterp(self.get_indices())
        outputs = outputs.reshape(outputs.shape[0], 1, outputs.shape[1])
        soc_hat = self.cnn(outputs)
        soc_hat = soc_hat.reshape(-1)
        return soc_hat

    def get_indices(self):
        return torch.sigmoid(self.indices)


class Algorithm_asa(Algorithm):
    def __init__(self, dataset, train_x, train_y, test_x, test_y, target_size, fold, reporter, verbose):
        super().__init__(dataset, train_x, train_y, test_x, test_y, target_size, fold, reporter, verbose)

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
        self.total_epoch = 700

        self.sae = SAE(self.train_x.shape[1],self.target_size)
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

        for epoch in range(self.total_epoch):
            optimizer_sae.zero_grad()
            x_hat = self.sae(self.train_x)
            loss = self.criterion_sae(x_hat, self.train_x)
            loss.backward()
            optimizer_sae.step()

        for epoch in range(self.total_epoch):
            optimizer_ann.zero_grad()
            hidden = self.sae.encoder(self.train_x)
            y_hat = self.ann(hidden)
            loss = self.criterion_ann(y_hat, self.train_y)
            loss.backward()
            optimizer_ann.step()
        return self

    def get_indices(self):
        indices = torch.round(self.ann.get_indices() * self.original_feature_size ).to(torch.int64).tolist()
        return list(dict.fromkeys(indices))
