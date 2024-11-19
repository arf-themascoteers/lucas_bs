import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


class DSManager:
    def __init__(self, name="lucas_r",folds=1):
        self.name = name
        dataset_path = f"data/{self.name}.csv"
        df = pd.read_csv(dataset_path)
        self.folds = folds
        df = df.sample(frac=1).reset_index(drop=True)
        self.data = df.to_numpy()

        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()

        X_scaled = scaler_X.fit_transform(self.data[:,0:-1])
        y_scaled = scaler_y.fit_transform(self.data[:,-1].reshape(-1,1)).ravel()

        self.scaler_y = scaler_y

    def get_k_folds(self):
        for i in range(self.folds):
            train_data, test_data = train_test_split(self.data, test_size=0.25, random_state=42+i)
            yield train_data[:,0:-1], train_data[:,-1], test_data[:,0:-1], test_data[:,-1]


if __name__ == '__main__':
    ds = DSManager(folds=1)
    for train_x, train_y, test_x, test_y in ds.get_k_folds():
        print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

    ds = DSManager(folds=4)
    for train_x, train_y, test_x, test_y in ds.get_k_folds():
        print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
