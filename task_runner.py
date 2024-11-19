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
                algorithm_object = Algorithm.create(algorithm, dataset, target_size, self.tag, self.reporter, self.verbose)
                algorithm.compute_performance(algorithm_object)


