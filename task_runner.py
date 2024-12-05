import torch
from ds_manager import DSManager
from reporter import Reporter
from algorithm import Algorithm


class TaskRunner:
    def __init__(self, task, folds=1,tag="results", verbose=False):
        torch.manual_seed(3)
        self.task = task
        self.folds = folds
        self.verbose = verbose
        self.tag = tag
        self.reporter = Reporter(self.tag)

    def evaluate(self):
        for index, algorithm in enumerate(self.task["algorithms"]):
            for dataset_name in self.task["datasets"]:
                for target_size in self.task["target_sizes"]:
                    for scale_y in self.task["scale_y"]:
                        dataset = DSManager(dataset_name, self.folds, scale_y)
                        for mode in self.task["mode"]:
                            for fold, (train_x, train_y, test_x, test_y) in enumerate(dataset.get_k_folds()):
                                if self.reporter.record_exists(algorithm, dataset_name, target_size, fold):
                                    print(algorithm, "for", dataset.name, "for target size", target_size,
                                          "for fold", fold, "for scale_y", scale_y, "for mode", mode,
                                          "was done. Skipping")
                                else:
                                    algorithm_object = Algorithm.create(algorithm, dataset, train_x, train_y,
                                                                        test_x,test_y, target_size, fold,
                                                                        scale_y, mode, self.reporter,self.verbose)
                                    algorithm_object.compute_fold()





