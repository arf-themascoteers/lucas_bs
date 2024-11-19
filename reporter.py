import os
import pandas as pd
import torch


class Reporter:
    def __init__(self, tag="results", skip_all_bands=False):
        self.tag = tag
        self.skip_all_bands = skip_all_bands
        self.subfolder = tag
        self.subfolder_path = os.path.join("results", self.subfolder)
        os.makedirs(self.subfolder_path, exist_ok=True)

        self.summary_filename = f"summary.csv"
        self.details_filename = f"details.csv"
        self.summary_file = os.path.join(self.subfolder_path, self.summary_filename)
        self.details_file = os.path.join(self.subfolder_path, self.details_filename)
        self.current_epoch_report_file = None

        if not os.path.exists(self.summary_file):
            with open(self.summary_file, 'w') as file:
                file.write("algorithm,dataset,target_size,r2,rmse,r2_train,rmse_train\n")

        if not os.path.exists(self.details_file):
            with open(self.details_file, 'w') as file:
                file.write("algorithm,dataset,target_size,r2,rmse,r2_train,rmse_train,fold,selected_bands\n")

    def get_summary(self):
        return self.summary_file

    def get_details(self):
        return self.details_file

    def update_summary(self, algorithm):
        if not self.record_exists(algorithm):

        df = pd.read_csv(self.details_file)
        df = df[
            (df['algorithm'] == algorithm.name) &
            (df['dataset'] == algorithm.dataset) &
            (df['target_size'] == algorithm.target_size)
            ]
        if df.empty:
            return

        result = df[df['algorithm'] == 'svr'].groupby('algorithm')[
            ['r2', 'rmse', 'r2_train', 'rmse_train']].mean().reset_index()
        result = result.iloc[0].to_dict()

        df = pd.read_csv(self.summary_file)
        filtered_df = df[df['algorithm'] == algorithm]
        if filtered_df.empty:
            df.loc[len(df)] = result

        df.loc[df['algorithm'] == 'svr', ['r2', 'rmse', 'r2_train', 'rmse_train']] = \
            [result["r2"],result["rmse"],result["r2_train"],result["rmse_train"]]

        df.to_csv(self.summary_file, index=False)

    def write_details(self, algorithm,r2,rmse,r2_train,rmse_train,fold):
        selected_bands = sorted(algorithm.selected_indices)
        with open(self.details_file, 'a') as file:
            file.write(f"{algorithm.get_name()},{algorithm.dataset},{algorithm.target_size},"
                       f"{r2},{rmse},{r2_train},{rmse_train},"
                       f"{fold},"
                       f"{'|'.join([str(i) for i in selected_bands])}\n")
        self.update_summary(algorithm)

    def record_exists(self, algorithm_object):
        algorithm = algorithm_object.name
        dataset = algorithm_object.dataset
        target_size = algorithm_object.target_size
        df = pd.read_csv(self.details_file)
        df = df[
            (df['algorithm'] == algorithm) &
            (df['dataset'] == dataset) &
            (df['target_size'] == target_size)
            ]
        if df.empty:
            return False
        return True

    @staticmethod
    def sanitize_metric(metric):
        if torch.is_tensor(metric):
            metric = metric.item()
        return round(max(metric, 0),3)

    def create_epoch_report(self, tag, algorithm, dataset, target_size):
        self.current_epoch_report_file = os.path.join("results", f"{tag}_{algorithm}_{dataset}_{target_size}.csv")

    def report_epoch_bsdr(self, epoch, mse_loss,oa,aa,k,selected_bands):
        if not os.path.exists(self.current_epoch_report_file):
            with open(self.current_epoch_report_file, 'w') as file:
                columns = ["epoch","loss","oa","aa","k"] + [f"band_{index+1}" for index in range(len(selected_bands))]
                file.write(",".join(columns)+"\n")

        with open(self.current_epoch_report_file, 'a') as file:
            file.write(f"{epoch},"
                       f"{Reporter.sanitize_metric(mse_loss)},"
                       f"{Reporter.sanitize_metric(oa)},{Reporter.sanitize_metric(aa)},{Reporter.sanitize_metric(k)},"
                       f"{','.join([str(i) for i in selected_bands])}\n"
                       )
