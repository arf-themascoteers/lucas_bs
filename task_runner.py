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
        for index, algorithm in enumerate(self.task["algorithms"]):
            for dataset_name in self.task["datasets"]:
                dataset = DSManager(name=dataset_name)
                for target_size in self.task["target_sizes"]:
                    for fold, (train_x, train_y, test_x, test_y) in enumerate(dataset.get_k_folds()):
                        if self.reporter.record_exists(algorithm, dataset, target_size, fold):
                            print(algorithm, "for", dataset, "for target size", target_size,"for fold", fold, "was done. Skipping")
                        else:
                            algorithm_object = Algorithm.create(algorithm, dataset_name, train_x, train_y,test_x,test_y, target_size, fold, self.reporter,self.verbose)
                            algorithm_object.compute_fold()





