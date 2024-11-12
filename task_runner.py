import torch
from ds_manager import DSManager
from reporter import Reporter
import pandas as pd
from algorithm import Algorithm


class TaskRunner:
    def __init__(self, task, tag="results", verbose=False):
        torch.manual_seed(3)
        self.task = task
        self.verbose = verbose
        self.tag = tag
        self.reporter = Reporter(self.tag)
        self.cache = pd.DataFrame(columns=["dataset","algorithm","props","cache_tag","oa","aa","k","time","selected_bands","selected_weights"])

    def evaluate(self):
        dataset = DSManager()
        for index, algorithm in enumerate(self.task["algorithms"]):
            for target_size in self.task["target_sizes"]:
                print(algorithm, target_size)
                algorithm_object = Algorithm.create(algorithm, target_size, dataset, self.tag, self.reporter, self.verbose)
                self.process_a_case(algorithm_object)

        self.reporter.save_results()
        return self.reporter.get_summary(), self.reporter.get_details()

    def process_a_case(self, algorithm:Algorithm):
        r2, rmse = self.reporter.get_saved_metrics(algorithm)
        if r2 is None:
            r2, rmse = algorithm.compute_performance()
            self.reporter.update_summary(algorithm, r2, rmse)
        else:
            print(algorithm.get_name(), "for", algorithm.dataset.get_name(), "for props", algorithm.get_props(), "for",
                  algorithm.target_size,"was done. Skipping")

    def get_results_for_a_case(self, algorithm:Algorithm):
        r2, rmse = algorithm.compute_performance()
        return r2, rmse

