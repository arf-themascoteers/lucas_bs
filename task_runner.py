import torch
from ds_manager import DSManager
from reporter import Reporter
from algorithm import Algorithm


class TaskRunner:
    def __init__(self, task, tag="results", verbose=False):
        torch.manual_seed(3)
        self.task = task
        self.verbose = verbose
        self.tag = tag
        self.reporter = Reporter(self.tag)

    def evaluate(self):
        dataset = DSManager()
        for index, algorithm in enumerate(self.task["algorithms"]):
            for target_size in self.task["target_sizes"]:
                print(algorithm, target_size)
                algorithm_object = Algorithm.create(algorithm, dataset, target_size, self.tag, self.reporter, self.verbose)
                self.process_a_case(algorithm_object)

        self.reporter.save_results()
        return self.reporter.get_summary(), self.reporter.get_details()

    def process_a_case(self, algorithm:Algorithm):
        if self.reporter.record_exists(algorithm):
            print(algorithm.get_name(), "for", algorithm.dataset.get_name(), "for",
                  algorithm.target_size,"was done. Skipping")
        else:
            r2, rmse = algorithm.compute_performance()
            self.reporter.write_details(algorithm, r2, rmse, r2_train, rmse_train, fold)


