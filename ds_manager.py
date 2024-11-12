import pandas as pd
from sklearn.model_selection import train_test_split


class DSManager:
    def __init__(self):
        dataset_path = f"data/lucas_r.csv"
        df = pd.read_csv(dataset_path)
        self.data = df.to_numpy()
        self.train_data, self.test_data = train_test_split(self.data, test_size=0.25, random_state=42)

    def get_train_x(self):
        return self.train_data[:,0:-1]

    def get_train_y(self):
        return self.train_data[:, -1]

    def get_test_x(self):
        return self.train_data[:,0:-1]

    def get_test_y(self):
        return self.train_data[:, -1]


