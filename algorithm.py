from abc import ABC, abstractmethod
from sklearn.metrics import r2_score, mean_squared_error
import torch
import importlib
import numpy as np


class Algorithm(ABC):
    def __init__(self, dataset,target_size,tag, reporter, verbose):
        self.dataset = dataset
        self.target_size = target_size
        self.tag = tag
        self.reporter = reporter
        self.verbose = verbose
        self.selected_indices = None
        self.weights = None
        self.all_indices = None
        self.reporter.create_epoch_report(tag, self.get_name(), self.dataset.get_name(), self.target_size)
        self.reporter.create_weight_report(tag, self.get_name(), self.dataset.get_name(), self.target_size)
        self.reporter.create_weight_all_report(tag, self.get_name(), self.dataset.get_name(), self.target_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _fitting_done(self):
        return self.selected_indices is not None

    def _fit(self):
        self.selected_indices = self.derive_selected_indices()
        return self.selected_indices

    def compute_performance(self):
        for fold, (train_x, test_x, train_y, test_y) in enumerate(self.dataset.get_k_folds()):
            if self.reporter.record_exists(self,fold):
                print(self.get_name(), "for", self.dataset.get_name(), "for", self.target_size, "for", fold, "was done. Skipping")
                continue
            else:
                train_y_hat = self.predict(train_x)
                test_y_hat = self.predict(test_x)
                train_r2, train_rmse = self.calculate_r2_rmse(train_y, train_y_hat)
                r2, rmse = self.calculate_r2_rmse(test_y, test_y_hat)
                self.reporter.write_details(self, r2, rmse, train_r2, train_rmse, fold)

    @staticmethod
    def calculate_r2_rmse(y_test, y_pred):
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = max(r2,0)
        rmse = max(rmse,0)
        return round(r2, 2), round(rmse, 2)


    @abstractmethod
    def derive_selected_indices(self):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def get_model(self):
        pass

    def get_name(self):
        class_name = self.__class__.__name__
        name_part = class_name[len("Algorithm_"):]
        return name_part

    @staticmethod
    def create(name, target_size, dataset, tag, reporter, verbose):
        class_name = f"Algorithm_{name}"
        module = importlib.import_module(f"algorithms.algorithm_{name}")
        clazz = getattr(module, class_name)
        return clazz(target_size, dataset, tag, reporter, verbose)
