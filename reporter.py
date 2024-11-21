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
                file.write("algorithm,dataset,target_size,r2,rmse,rpd,rpiq,train_r2,train_rmse,train_rpd,train_rpiq\n")

        if not os.path.exists(self.details_file):
            with open(self.details_file, 'w') as file:
                file.write("algorithm,dataset,target_size,r2,rmse,rpd,rpiq,train_r2,train_rmse,train_rpd,train_rpiq,fold,selected_bands\n")

        self.current_epoch_report_file = None

    def get_summary(self):
        return self.summary_file

    def get_details(self):
        return self.details_file

    def update_summary(self, algorithm, dataset, target_size):
        df = pd.read_csv(self.details_file)
        df = df[
            (df['algorithm'] == algorithm) &
            (df['dataset'] == dataset) &
            (df['target_size'] == target_size)
            ]
        if df.empty:
            return

        average = df[['r2', 'rmse', 'rpd','rpiq','train_r2', 'train_rmse','train_rpd','train_rpiq']].mean()

        summary_df_original = pd.read_csv(self.summary_file)

        summary_df = summary_df_original[
            (summary_df_original['algorithm'] == algorithm) &
            (summary_df_original['dataset'] == dataset) &
            (summary_df_original['target_size'] == target_size)
            ]
        if summary_df.empty:
            summary_df_original.loc[len(summary_df_original)] = {
                "algorithm":algorithm,
                "dataset":dataset,
                "target_size":target_size,
                "r2" : average['r2'],
                "rmse" : average['rmse'],
                "rpd" : average['rpd'],
                "rpiq" : average['rpiq'],
                "train_r2" : average['train_r2'],
                "train_rmse" : average['train_rmse'],
                "train_rpd" : average['train_rpd'],
                "train_rpiq" : average['train_rpiq']
            }
            summary_df_original.to_csv(self.summary_file, index=False)
        else:
            summary_df_original.loc[
                (summary_df_original['algorithm'] == algorithm) &
                (summary_df_original['dataset'] == dataset) &
                (summary_df_original['target_size'] == target_size)
                ,
                ["r2","rmse","rpd","rpiq","train_r2","train_rmse","train_rpd","train_rpiq"]
            ] = [average['r2'],average['rmse'],average['rpd'],average['rpiq'],average['train_r2'],average['train_rmse'],average['rpd'],average['rpiq']]
            summary_df_original.to_csv(self.summary_file, index=False)

    def write_details(self, algorithm,dataset, target_size, r2,rmse,rpd,rpiq,train_r2,train_rmse,train_rpd,train_rpiq,fold,selected_bands):
        selected_bands = sorted(selected_bands)
        with open(self.details_file, 'a') as file:
            file.write(f"{algorithm},{dataset},{target_size},"
                       f"{r2},{rmse},{rpd},{rpiq},"
                       f"{train_r2},{train_rmse},{train_rpd},{train_rpiq},"
                       f"{fold},"
                       f"{'|'.join([str(i) for i in selected_bands])}\n")
        self.update_summary(algorithm,dataset, target_size)

    def record_exists(self, algorithm, dataset, target_size, fold):
        df = pd.read_csv(self.details_file)
        df = df[
            (df['algorithm'] == algorithm) &
            (df['dataset'] == dataset) &
            (df['target_size'] == target_size) &
            (df['fold'] == fold)
            ]
        if df.empty:
            return False
        return True

    @staticmethod
    def sanitize_metric(metric):
        if torch.is_tensor(metric):
            metric = metric.item()
        metric = max(0,metric)
        return round(metric,5)

    def create_epoch_report(self, algorithm, dataset, target_size, fold):
        self.current_epoch_report_file = os.path.join(self.subfolder_path, f"{algorithm}_{dataset}_{target_size}_{fold}.csv")

    def report_epoch_bsdr(self, epoch, r2, rmse, rpd, rpiq, train_r2, train_rmse, train_rpd, train_rpiq, selected_bands):
        if not os.path.exists(self.current_epoch_report_file):
            with open(self.current_epoch_report_file, 'w') as file:
                columns = ["epoch","r2","rmse","rpd","rpiq","train_r2","train_rmse","train_rpd","train_rpiq"] + [f"band_{index+1}" for index in range(len(selected_bands))]
                file.write(",".join(columns)+"\n")

        with open(self.current_epoch_report_file, 'a') as file:
            file.write(f"{epoch},"
                       f"{Reporter.sanitize_metric(r2)},"
                       f"{Reporter.sanitize_metric(rmse)},"
                       f"{Reporter.sanitize_metric(rpd)},"
                       f"{Reporter.sanitize_metric(rpiq)},"
                       f"{Reporter.sanitize_metric(train_r2)},"
                       f"{Reporter.sanitize_metric(train_rmse)},"
                       f"{Reporter.sanitize_metric(train_rpd)},"
                       f"{Reporter.sanitize_metric(train_rpiq)},"
                       f"{','.join([str(i) for i in selected_bands])}\n"
                       )
